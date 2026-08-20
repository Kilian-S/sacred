"""SMDP decision wrapper for SACRED training.

The low-level :class:`GraphEnv` ticks every simulated second. This wrapper
skips non-decision ticks and yields only meaningful decision events for the
protagonist or antagonist, while accumulating rewards and elapsed time for
SMDP replay records.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Callable, Mapping

import networkx as nx

from src.env.graph_env import EdgeId, GraphEnv, NodeId, StepResult
from src.env.toy_graph import make_toy_graph_env


class DecisionType(str, Enum):
    """Decision event types exposed by the SMDP wrapper."""

    PROTAGONIST_DECISION = "protagonist_decision"
    ANTAGONIST_DECISION = "antagonist_decision"
    BOTH_DECISION = "both_decision"
    TERMINAL = "terminal"


@dataclass(frozen=True, slots=True)
class SMDPConfig:
    """Configuration for event-driven SACRED training."""

    tick_seconds: float = 1.0
    max_ticks: int = 240
    antagonist_interval: int = 20
    congestion_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    default_congestion_level: float = 0.65
    congestion_duration: int = 40
    congestion_budget: float = 120.0
    congestion_cooldown: int = 3
    delivery_reward: float = 0.0
    time_penalty: float = 1.0
    remaining_demand_penalty: float = 0.5
    congestion_cost: float = 0.02
    # Reward shaping selector. "legacy" = the static delivery/time/remaining-demand shaping above.
    # "latency" = per tick, protagonist_reward = -(outstanding arrived-but-undelivered request
    # count), which telescopes over the episode to the total delivery latency (sum of per-request
    # waits).
    reward_mode: str = "legacy"
    # Protagonist action model. "destination" = pick a target node; env routes there via A*.
    # "next_hop" = pick the next adjacent node, one edge at a time, so the policy chooses the
    # route and the antagonist's congestion becomes something it can route around. See
    # dispatch_truck_edge in graph_env.
    routing_mode: str = "destination"
    # Next-hop only: a next-hop is allowed if the route through it stays within this factor of
    # the shortest static distance to the goal. Keeps near-shortest alternatives (the safe route)
    # in the choice set while excluding detours/backtracking that would let the truck wander the
    # whole map. 1.5 keeps any route up to 50% longer than optimal.
    routing_corridor_slack: float = 1.5
    # Antagonist sequential-epoch cap: max congestion sub-actions per decision event (0 =
    # unlimited). Capping to 1 makes the adversary place one strategic roadblock per event
    # (reactive; sustained via congestion_duration) instead of chaining many sub-actions within
    # a single event.
    max_antag_actions_per_event: int = 0
    # Antagonist reach: which edges it may block. "leashed" (default) = the 3-hop radius around
    # trucks (reactive: block right in front). "route" = the edges on each truck's shortest path
    # to its target, so it can pre-block the gateway ahead of the truck (anticipation).
    antag_reach: str = "leashed"
    # Reward baseline. "none" (default) = the raw latency reward. "twin" = subtract a per-tick
    # action-independent baseline b(t) from the latency reward, so protagonist_reward =
    # -(remaining_demand - b(t)); b(t) is supplied by a baseline_provider injected into the
    # wrapper. Since b(t) does not depend on either agent's in-episode actions, the episode
    # reward telescopes to total_wait - sum_t b(t): the zero-sum game and its equilibrium are
    # preserved (antagonist_reward = -protagonist_reward throughout), only the gradient variance
    # changes.
    reward_baseline: str = "none"


@dataclass(slots=True)
class SMDPTransition:
    """Replay-buffer-ready transition with SMDP elapsed time."""

    agent: str
    state: dict[str, Any]
    action: Any
    reward: float
    next_state: dict[str, Any]
    done: bool
    elapsed_ticks: int
    action_mask: dict[str, Any]
    info: dict[str, Any] = field(default_factory=dict)
    # Lazily-populated cache of featurized graphs (keyed "state"/"next"), filled on first use
    # and reused thereafter. featurize_state is a pure function of the (immutable, buffered)
    # state, so this is behaviour-preserving. Freed with the transition when it is evicted
    # from the replay buffer.
    feature_cache: dict[str, Any] = field(default_factory=dict)

    def __getstate__(self) -> dict[str, Any]:
        # Exclude the featurization cache from pickling so serialised replay-buffer checkpoints
        # stay small. The live in-memory cache is untouched; it is rebuilt lazily after a resume.
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "feature_cache"}

    def __setstate__(self, state: Any) -> None:
        # `state` is a plain dict (current format) or a `(None, slot_dict)` tuple (older
        # pickles of this slotted dataclass). Handle both.
        if isinstance(state, tuple):
            state = state[1] or {}
        for key, value in state.items():
            setattr(self, key, value)
        self.feature_cache = {}


@dataclass(slots=True)
class DecisionEvent:
    """Decision event returned by the SMDP wrapper."""

    decision_type: DecisionType
    observation: dict[str, Any]
    waiting_trucks: list[int]
    protagonist_reward: float
    antagonist_reward: float
    elapsed_ticks: int
    protagonist_action_mask: dict[int, list[NodeId]]
    antagonist_action_mask: dict[str, Any]
    done: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class CongestionBudget:
    """Budget accounting for environment-altering antagonist actions."""

    total: float
    used: float = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.used)

    def spend(self, amount: float) -> bool:
        if amount > self.remaining + 1e-12:
            return False
        self.used += amount
        return True


class SMDPDecisionWrapper:
    """Event-driven wrapper around one-second graph physics."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], GraphEnv] = make_toy_graph_env,
        config: SMDPConfig | None = None,
        baseline_provider: Callable[[GraphEnv], tuple[dict[int, float], float]] | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.config = config or SMDPConfig()
        self.env = self.env_factory()
        # Reward baseline (see SMDPConfig.reward_baseline). `baseline_provider(env) ->
        # (series, last)` maps post-step env.time -> b(t) plus a pad value for ticks past the
        # provider's range; called once per reset when reward_baseline != "none". `_baseline_record`
        # is a recording buffer used to capture a series during a provider's own rollout (set
        # externally, never by reset).
        self._baseline_provider = baseline_provider
        self._reward_baseline_series: dict[int, float] | None = None
        self._reward_baseline_last: float = 0.0
        self._baseline_record: list[tuple[int, float]] | None = None
        self.budget = CongestionBudget(self.config.congestion_budget)
        self.active_congestion: dict[EdgeId, int] = {}
        self.congestion_heap: list[tuple[int, EdgeId]] = []
        self.cooldown_remaining = 0
        self.next_antagonist_tick = self.config.antagonist_interval
        self._last_decision_observation: dict[str, Any] | None = None
        self._last_action: Any = None
        self._last_action_mask: dict[str, Any] = {}
        self._last_agent: str | None = None
        self._accumulated_protagonist_reward = 0.0
        self._accumulated_antagonist_reward = 0.0
        self._elapsed_ticks = 0
        self._internal_events: list[dict[str, Any]] = []
        self._in_sequential_epoch = False
        self._antag_epoch_actions = 0  # sub-actions taken in the current antagonist decision epoch
        # Cache of static (congestion-free) single-source distances per goal node, used by the
        # next-hop forward mask. Topology is fixed within an episode, so this is rebuilt at reset.
        self._goal_dist_cache: dict[NodeId, dict[NodeId, float]] = {}

    def reset_decision_env(self) -> DecisionEvent:
        """Reset and return the first decision event."""

        self.env = self.env_factory()
        self.env.max_time = self.config.max_ticks
        self.env.reset()
        self.budget = CongestionBudget(self.config.congestion_budget)
        self.active_congestion = {}
        self.congestion_heap = []
        self.cooldown_remaining = 0
        self.next_antagonist_tick = self.config.antagonist_interval
        self._last_decision_observation = None
        self._last_action = None
        self._last_action_mask = {}
        self._last_agent = None
        self._reset_accumulators()
        self._in_sequential_epoch = False
        self._antag_epoch_actions = 0
        self._goal_dist_cache = {}
        # Compute this episode's action-independent baseline from the freshly-generated arrivals
        # (the provider replays them clean under greedy). Skipped when disabled or when this
        # wrapper is itself a provider's twin (no provider injected). `_baseline_record` is left
        # untouched so a provider can capture a series across this reset.
        if self.config.reward_baseline == "twin" and self._baseline_provider is not None:
            self._reward_baseline_series, self._reward_baseline_last = \
                self._baseline_provider(self.env)
        if self.config.routing_mode in ("next_hop", "hybrid"):
            # Auto-resolve any forced moves to the first real (>=2-option) decision.
            return self.advance_until_decision()
        return self._build_event(DecisionType.PROTAGONIST_DECISION)

    def step_protagonist(self, actions_by_truck: Mapping[int, NodeId]) -> tuple[DecisionEvent, SMDPTransition]:
        """Apply protagonist decisions and advance to the next event."""

        state = self.env.observe()
        action_mask = self.protagonist_action_mask()
        dispatch_actions = self._valid_protagonist_actions(actions_by_truck, action_mask)
        if self.config.routing_mode == "next_hop":
            step_result = self.env.step(next_hop_dispatch=dispatch_actions)
        elif self.config.routing_mode == "hybrid":
            # Split by decision type: an unassigned truck's action is an ASSIGNMENT (set the target,
            # no movement); an assigned truck's action is a ROUTING next-hop (move one edge).
            routing: dict[int, NodeId] = {}
            for tid, node in dispatch_actions.items():
                truck = self.env.trucks[tid]
                if truck.assigned_target is None:
                    truck.assigned_target = node
                else:
                    routing[tid] = node
            step_result = self.env.step(next_hop_dispatch=routing)
        else:
            step_result = self.env.step(dispatch_actions=dispatch_actions)
        self._accumulate_step(step_result, antagonist_action={})
        next_event = self.advance_until_decision()
        transition = SMDPTransition(
            agent="protagonist",
            state=state,
            action=dict(dispatch_actions),
            reward=next_event.protagonist_reward,
            next_state=next_event.observation,
            done=next_event.done,
            elapsed_ticks=next_event.elapsed_ticks,
            action_mask={"protagonist": action_mask},
            info=dict(next_event.info),
        )
        return next_event, transition

    def step_antagonist(self, action: tuple[EdgeId, float] | None) -> tuple[DecisionEvent, SMDPTransition]:
        """Apply one antagonist action and advance to the next decision event."""

        state = self.env.observe()
        action_mask = self.antagonist_action_mask()
        congestion_action = self._valid_antagonist_action(action, action_mask)

        applied_any = False
        congestion_spend = 0.0
        for edge, level in congestion_action.items():
            applied_any = True
            congestion_spend += level * self.config.congestion_duration
            self.env.set_congestion(edge, level)
            self._internal_events.append(
                {
                    "time": self.env.time,
                    "dispatched": [],
                    "ignored_dispatches": [],
                    "congestion_updates": [{"edge": edge, "congestion_level": level}],
                    "arrivals": [],
                    "deliveries": [],
                    "reloads": [],
                    "distance_travelled": 0.0,
                }
            )
            
            # GPS-style dynamic rerouting for trucks affected by this newly injected congestion.
            # Only in destination mode: in next_hop mode the policy owns rerouting, so
            # auto-rerouting would erase the decision the protagonist must learn.
            if level > 0.0 and self.config.routing_mode == "destination":
                for truck_id, truck in self.env.trucks.items():
                    if truck.is_idle or not truck.path:
                        continue
                    if edge in getattr(truck, 'path_edges', set()):
                        k = truck.path_index
                        has_edge = False
                        for i in range(k, len(truck.path) - 1):
                            u = truck.path[i]
                            v = truck.path[i+1]
                            if self.env._edge_key(u, v) == edge or self.env._edge_key(v, u) == edge:
                                has_edge = True
                                break
                        if has_edge:
                            truncate_idx = max(k + 2, i + 1)
                            truck.path = list(truck.path[:truncate_idx])
                            truck.destination = truck.path[-1]

        if applied_any:
            self._antag_epoch_actions += 1

        # Determine if we continue the sequential decision loop
        min_cost = min(self.config.congestion_levels) * self.config.congestion_duration

        # Temporarily enable sequential epoch flag to generate mask ignoring cooldown
        original_seq_flag = getattr(self, "_in_sequential_epoch", False)
        self._in_sequential_epoch = True
        next_mask = self.antagonist_action_mask()
        has_allowed_edges = any(next_mask.get("levels_by_edge", {}).values())
        self._in_sequential_epoch = original_seq_flag

        cap = self.config.max_antag_actions_per_event
        under_cap = (cap == 0) or (self._antag_epoch_actions < cap)
        continue_loop = (
            action is not None
            and applied_any
            and self.budget.remaining >= min_cost - 1e-6
            and has_allowed_edges
            and under_cap
        )

        # Apply the congestion penalty
        cost_penalty = self.config.congestion_cost * congestion_spend
        self._accumulated_antagonist_reward -= cost_penalty

        if continue_loop:
            # Stay in sequential decision epoch (elapsed_ticks = 0, no simulated time progression)
            self._in_sequential_epoch = True
            next_event = self._build_event(DecisionType.ANTAGONIST_DECISION)
            transition = SMDPTransition(
                agent="antagonist",
                state=state,
                action=action,
                reward=next_event.antagonist_reward,
                next_state=next_event.observation,
                done=next_event.done,
                elapsed_ticks=0,
                action_mask={"antagonist": action_mask},
                info=dict(next_event.info),
            )
            return next_event, transition
        else:
            # Terminate sequential epoch: advance simulated time until the next decision event
            self._in_sequential_epoch = False
            self._antag_epoch_actions = 0
            self.next_antagonist_tick += self.config.antagonist_interval
            next_event = self.advance_until_decision()
            transition = SMDPTransition(
                agent="antagonist",
                state=state,
                action=action,
                reward=next_event.antagonist_reward,
                next_state=next_event.observation,
                done=next_event.done,
                elapsed_ticks=next_event.elapsed_ticks,
                action_mask={"antagonist": action_mask},
                info=dict(next_event.info),
            )
            return next_event, transition

    def advance_until_decision(self) -> DecisionEvent:
        """Run internal one-second ticks until a decision event is reached."""

        self._auto_resolve_forced_moves()
        immediate = self._current_decision_type()
        if immediate is not None:
            return self._build_event(immediate)

        while True:
            step_result = self.env.step()
            self._accumulate_step(step_result, antagonist_action={})
            self._age_congestion()
            self._auto_resolve_forced_moves()

            decision_type = self._current_decision_type()
            if decision_type is not None:
                return self._build_event(decision_type)

    def protagonist_action_mask(self) -> dict[int, list[NodeId]]:
        """Return valid next-node actions for all waiting trucks."""

        if not any(t.is_idle for t in self.env.trucks.values()):
            return {}

        if self.config.routing_mode == "next_hop":
            return self._next_hop_action_mask()
        if self.config.routing_mode == "hybrid":
            return self._hybrid_action_mask()

        mask: dict[int, list[NodeId]] = {}

        # Precompute total targeted loads for each destination
        targeted_loads: dict[NodeId, float] = {}
        for t in self.env.trucks.values():
            if t.destination is not None:
                targeted_loads[t.destination] = targeted_loads.get(t.destination, 0.0) + t.load

        valid_customers_by_comp = getattr(self.env, 'valid_customers_by_comp', {})

        for truck_id, truck in self.env.trucks.items():
            if not truck.is_idle:
                continue

            current_node = truck.current_node
            if current_node is None:
                continue

            truck_comp = self.env.node_to_component.get(current_node)
            depot_comp = self.env.node_to_component.get(truck.home_depot)
            can_reach_depot = (truck_comp == depot_comp)

            if truck.load <= 0 and current_node != truck.home_depot:
                mask[truck_id] = [truck.home_depot] if can_reach_depot else []
            elif truck.load <= 0:
                mask[truck_id] = []
            elif self._remaining_demand() <= 0 and current_node != truck.home_depot:
                mask[truck_id] = [truck.home_depot] if can_reach_depot else []
            elif self._remaining_demand() <= 0:
                mask[truck_id] = []
            else:
                # Goal-Directed: All customer nodes with positive unassigned demand, plus the depot
                destinations = []
                for n, node_demand in valid_customers_by_comp.get(truck_comp, {}).items():
                    # Subtract in-transit/committed load of all OTHER trucks targeting this node
                    other_targeted = targeted_loads.get(n, 0.0)
                    if truck.destination == n:
                        other_targeted -= truck.load
                    unassigned_demand = node_demand - other_targeted
                    if unassigned_demand > 0.0:
                        destinations.append(n)
                
                # Only allow depot if load < capacity, or if the truck is not already at the depot.
                if truck.load < truck.capacity or current_node != truck.home_depot:
                    if can_reach_depot:
                        destinations.append(truck.home_depot)
                
                # Filter out the current node (no choosing to stay at current customer)
                if current_node in destinations and current_node != truck.home_depot:
                    destinations.remove(current_node)
                mask[truck_id] = destinations
        return mask

    def _next_hop_action_mask(self) -> dict[int, list[NodeId]]:
        """Next-hop routing: each idle truck may step to a forward neighbour, one that makes
        topological progress toward its goal (nearest demand if loaded, else the home depot),
        by static congestion-free distance.

        Bounds exploration to a corridor while preserving route choice: at a branch, both the
        fast and the safe route are forward, so both appear and the policy decides between them.
        Congestion is read from the observation, not the mask, so the antagonist can still make
        the fast route costly. Falls back to all neighbours if none is strictly closer.
        """
        mask: dict[int, list[NodeId]] = {}
        for truck_id, truck in self.env.trucks.items():
            if not truck.is_idle or truck.current_node is None:
                continue
            mask[truck_id] = self._forward_mask(truck, self._truck_goal(truck))
        return mask

    def _forward_mask(self, truck: Any, goal: NodeId | None) -> list[NodeId]:
        """Forward next-hop neighbours toward ``goal`` (the corridor mask) with anti-oscillation.
        Shared by next-hop and hybrid routing (hybrid passes the truck's ``assigned_target``)."""
        neighbors = sorted(self.env.graph.neighbors(truck.current_node), key=repr)
        if goal is None or goal == truck.current_node:
            return neighbors
        dist_to_goal = self._dist_to_goal(goal)
        cur = dist_to_goal.get(truck.current_node, float("inf"))
        budget = self.config.routing_corridor_slack * cur + 1e-9
        forward = [
            n for n in neighbors
            if self.env.graph.edges[truck.current_node, n]["distance"] + dist_to_goal.get(n, float("inf")) <= budget
        ]
        forward = forward if forward else neighbors
        # Anti-oscillation: drop the node we just came from *only* when it is a backward move (not
        # strictly closer to the goal). Allows a legitimate turnaround when the goal flips to depot.
        if truck.path_index >= 1 and len(truck.path) > truck.path_index:
            prev = truck.path[truck.path_index - 1]
            if dist_to_goal.get(prev, float("inf")) >= cur:
                pruned = [n for n in forward if n != prev]
                if pruned:
                    forward = pruned
        return forward

    def _hybrid_action_mask(self) -> dict[int, list[NodeId]]:
        """Hybrid (assignment + next-hop routing). Per idle truck, the decision type depends on its
        state: **no assigned target + load** -> ASSIGNMENT (pick a pending request); **has an assigned
        target** (a request, or home_depot on the return leg) -> ROUTING (forward next-hop toward it).
        ``assigned_target`` is managed by the env on serve/reload."""
        mask: dict[int, list[NodeId]] = {}
        for truck_id, truck in self.env.trucks.items():
            if not truck.is_idle or truck.current_node is None:
                continue
            if truck.assigned_target is not None:
                mask[truck_id] = self._forward_mask(truck, truck.assigned_target)
            elif truck.load > 0:
                mask[truck_id] = self._assignment_candidates(truck)
            else:  # empty + unassigned (rare) -> route home to reload
                mask[truck_id] = self._forward_mask(truck, truck.home_depot)
        return mask

    def _assignment_candidates(self, truck: Any) -> list[NodeId]:
        """Pending request nodes reachable from the truck (same component), excluding requests
        another truck is already assigned to (cross-event claiming; same-event claiming is still
        applied by the trainer/eval). Prevents a request being double-assigned across decision
        events. Falls back to the truck's own depot if no unclaimed request remains, so the
        episode can terminate; the env clears the assignment on depot arrival, so it re-enters
        assignment if new work appears."""
        comp = self.env.node_to_component.get(truck.current_node)
        customers = getattr(self.env, "valid_customers_by_comp", {}).get(comp, {})
        taken = {
            t.assigned_target
            for t in self.env.trucks.values()
            if t is not truck and t.assigned_target is not None
        }
        candidates = [
            n for n in sorted(customers, key=repr)
            if n != truck.current_node and n not in taken
        ]
        if candidates:
            return candidates
        if truck.home_depot is not None and truck.current_node != truck.home_depot:
            return [truck.home_depot]
        return []

    def _truck_goal(self, truck: Any) -> NodeId | None:
        """The node a truck is heading for: nearest demand if it has load, else home depot."""
        if truck.load > 0:
            customers = self._customers_with_demand()
            if customers:
                return min(
                    customers,
                    key=lambda c: (self._dist_to_goal(c).get(truck.current_node, float("inf")), repr(c)),
                )
        return truck.home_depot

    def _customers_with_demand(self) -> list[NodeId]:
        valid_customers_by_comp = getattr(self.env, "valid_customers_by_comp", None)
        if valid_customers_by_comp is not None:
            return [n for customers in valid_customers_by_comp.values() for n in customers]
        return [
            n for n, data in self.env.graph.nodes(data=True)
            if not data.get("has_depot", False) and data.get("demand", 0.0) > 0.0
        ]

    def _dist_to_goal(self, goal: NodeId) -> dict[NodeId, float]:
        cache = self._goal_dist_cache
        if goal not in cache:
            cache[goal] = nx.single_source_dijkstra_path_length(self.env.graph, goal, weight="distance")
        return cache[goal]

    def _auto_resolve_forced_moves(self) -> None:
        """Dispatch idle trucks with exactly one forward ROUTING option, so a protagonist decision is
        only surfaced at a genuine (>=2-option) branch. Assignment decisions are never auto-resolved."""
        mode = self.config.routing_mode
        if mode == "next_hop":
            for truck_id, options in self._next_hop_action_mask().items():
                if len(options) == 1:
                    self.env.dispatch_truck_edge(truck_id, options[0])
        elif mode == "hybrid":
            for truck_id, truck in self.env.trucks.items():
                if truck.assigned_target is None or not truck.is_idle or truck.current_node is None:
                    continue  # only routing (already-assigned) trucks auto-resolve
                options = self._forward_mask(truck, truck.assigned_target)
                if len(options) == 1:
                    self.env.dispatch_truck_edge(truck_id, options[0])

    def _route_reach_edges(self) -> set:
        """Route-reach: the edges on each truck's static shortest path to its target, letting the
        adversary pre-block the gateway ahead on a truck's committed route (anticipation). Static
        (congestion-free) distance is the truck's intended route, so a block forces the truck to
        react (detour) rather than the adversary chasing the reroute. A truck with no target (e.g.
        at a depot awaiting assignment) is not yet committed and contributes no edges."""
        edges: set = set()
        for truck in self.env.trucks.values():
            target = getattr(truck, "assigned_target", None) or truck.destination
            if target is None:
                continue
            start = truck.current_node
            if start is None and truck.edge is not None:
                start = truck.edge[1]  # heading toward the far endpoint of the current edge
            if start is None or start == target:
                continue
            try:
                path = nx.shortest_path(self.env.graph, start, target, weight="distance")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for i in range(len(path) - 1):
                edges.add(self.env._edge_key(path[i], path[i + 1]))
        return edges

    def antagonist_action_mask(self) -> dict[str, Any]:
        """Return the antagonist's valid action choices (edges and levels) at the current event."""

        if not getattr(self, "_in_sequential_epoch", False) and self.cooldown_remaining > 0:
            return {"can_wait": True, "levels_by_edge": {}}

        # 1. Action-space masking: which edges the antagonist may block (its reach).
        if self.config.antag_reach == "route":
            nearby_edges = self._route_reach_edges()
        else:
            # leashed: the 3-hop radius around trucks (O(1) via precomputed k-hop sets)
            nearby_edges = set()
            for truck in self.env.trucks.values():
                if truck.current_node is not None:
                    nearby_edges.update(self.env._k_hop_edges[truck.current_node])
                elif truck.edge is not None:
                    u, v = truck.edge
                    nearby_edges.update(self.env._k_hop_edges[u])
                    nearby_edges.update(self.env._k_hop_edges[v])

        # 2. Filter allowed levels by mask and remaining budget
        levels_by_edge: dict[EdgeId, list[float]] = {}
        for edge in sorted(list(nearby_edges)):
            if edge in self.active_congestion:
                continue
            valid_levels = [
                level
                for level in self.config.congestion_levels
                if (level * self.config.congestion_duration) <= self.budget.remaining + 1e-12
            ]
            if valid_levels:
                levels_by_edge[edge] = valid_levels
        return {"can_wait": True, "levels_by_edge": levels_by_edge}



    def _current_decision_type(self) -> DecisionType | None:
        if self.env.is_done():
            return DecisionType.TERMINAL

        mask = self.protagonist_action_mask()
        if self.config.routing_mode == "next_hop":
            # Only a >=2-option node is a real decision; forced 1-option moves are auto-resolved.
            protagonist_due = any(len(options) >= 2 for options in mask.values())
        elif self.config.routing_mode == "hybrid":
            # Assignment (unassigned truck) is due at >=1 candidate; routing (assigned truck) only
            # at a >=2-option branch (1-option routing moves are auto-resolved).
            protagonist_due = False
            for truck_id, options in mask.items():
                if self.env.trucks[truck_id].assigned_target is None:
                    protagonist_due = protagonist_due or bool(options)
                else:
                    protagonist_due = protagonist_due or len(options) >= 2
        else:
            protagonist_due = any(options for options in mask.values())
        antagonist_due = self.env.time >= self.next_antagonist_tick

        if protagonist_due and antagonist_due:
            return DecisionType.BOTH_DECISION
        if antagonist_due:
            return DecisionType.ANTAGONIST_DECISION
        if protagonist_due:
            return DecisionType.PROTAGONIST_DECISION
        return None

    def _build_event(self, decision_type: DecisionType) -> DecisionEvent:
        prot_mask = self.protagonist_action_mask()
        event = DecisionEvent(
            decision_type=decision_type,
            observation=self.env.observe(),
            waiting_trucks=[truck_id for truck_id, options in prot_mask.items() if options],
            protagonist_reward=self._accumulated_protagonist_reward,
            antagonist_reward=self._accumulated_antagonist_reward,
            elapsed_ticks=self._elapsed_ticks,
            protagonist_action_mask=prot_mask,
            antagonist_action_mask=self.antagonist_action_mask(),
            done=decision_type == DecisionType.TERMINAL,
            info={
                "events": list(self._internal_events),
                "budget_used": self.budget.used,
                "budget_remaining": self.budget.remaining,
                "active_congestion": dict(self.active_congestion),
            },
        )
        self._reset_accumulators()
        return event

    def _valid_protagonist_actions(
        self,
        actions_by_truck: Mapping[int, NodeId],
        action_mask: Mapping[int, list[NodeId]],
    ) -> dict[int, NodeId]:
        valid: dict[int, NodeId] = {}
        for truck_id, destination in actions_by_truck.items():
            if destination in action_mask.get(truck_id, []):
                valid[truck_id] = destination
        return valid

    def _valid_antagonist_action(
        self,
        action: tuple[EdgeId, float] | None,
        action_mask: Mapping[str, Any],
    ) -> dict[EdgeId, float]:
        if action is None:
            return {}

        edge, raw_level = action
        level = float(raw_level)
        levels_by_edge = action_mask.get("levels_by_edge", {})
        if level not in levels_by_edge.get(edge, []):
            return {}

        cost = level * self.config.congestion_duration
        if not self.budget.spend(cost):
            return {}

        self.active_congestion[edge] = self.env.time + self.config.congestion_duration
        heapq.heappush(self.congestion_heap, (self.active_congestion[edge], edge))
        self.cooldown_remaining = self.config.congestion_cooldown
        return {edge: level}

    def _accumulate_step(self, step_result: StepResult, antagonist_action: Mapping[EdgeId, float]) -> None:
        if self.config.reward_mode == "latency":
            # Delivery-latency objective: pay -1 per tick per outstanding demand unit (arrived but
            # not yet delivered). Measured after this tick's deliveries (step_result is post-step),
            # so completing a unit immediately stops accruing its penalty. Summed over the episode
            # this telescopes to total latency, i.e. sum over units of (delivery_tick -
            # arrival_tick). Using units (not nodes) keeps it correct when a single node holds
            # many units (the next-hop target).
            remaining = float(self.env.remaining_demand)
            # Recording pass: capture this tick's remaining_demand for a provider's twin series.
            if self._baseline_record is not None:
                self._baseline_record.append((int(self.env.time), remaining))
            # Subtraction pass: strip the action-independent baseline b(t). Ticks past the
            # provider's range (the real episode ran longer under attack than the clean twin) pad
            # with the twin's final value. b(t) is constant in both agents' actions, so the game
            # stays zero-sum and the telescoped total shifts by exactly sum_t b(t) (a per-episode
            # constant).
            if self._reward_baseline_series is not None:
                remaining = remaining - self._reward_baseline_series.get(
                    int(self.env.time), self._reward_baseline_last)
            protagonist_reward = -remaining
        else:
            delivered = sum(delivery["delivered"] for delivery in step_result.info["deliveries"])
            remaining_demand = self._remaining_demand()
            protagonist_reward = (
                (self.config.delivery_reward * delivered)
                - self.config.time_penalty
                - (self.config.remaining_demand_penalty * remaining_demand)
            )
        antagonist_reward = -protagonist_reward

        self._accumulated_protagonist_reward += protagonist_reward
        self._accumulated_antagonist_reward += antagonist_reward
        self._elapsed_ticks += 1
        self._internal_events.append(step_result.info)

    def _age_congestion(self) -> None:
        while self.congestion_heap and self.congestion_heap[0][0] <= self.env.time:
            expiration_tick, edge = heapq.heappop(self.congestion_heap)
            if self.active_congestion.get(edge) == expiration_tick:
                self.active_congestion.pop(edge, None)
                if self.env.graph.has_edge(*edge):
                    self.env.set_congestion(edge, 0.0)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def _remaining_demand(self) -> float:
        return self.env.remaining_demand

    def _outstanding_requests(self) -> int:
        """Count requests that have arrived but are not yet delivered.

        When all requests are present from t=0, this is simply the number of non-depot nodes
        with positive remaining demand. Uses ``valid_customers_by_comp`` when available (O(1)
        per tick), else falls back to a graph scan.
        """
        valid_customers_by_comp = getattr(self.env, "valid_customers_by_comp", None)
        if valid_customers_by_comp is not None:
            return sum(len(customers) for customers in valid_customers_by_comp.values())
        return sum(
            1
            for _, data in self.env.graph.nodes(data=True)
            if not data.get("has_depot", False) and data.get("demand", 0.0) > 0.0
        )

    def _neighbors_toward(self, source: NodeId, target: NodeId) -> list[NodeId]:
        path = self.env._get_shortest_path(source, target)
        if not path or len(path) < 2:
            return []
        return [path[1]]

    def _reset_accumulators(self) -> None:
        self._accumulated_protagonist_reward = 0.0
        self._accumulated_antagonist_reward = 0.0
        self._elapsed_ticks = 0
        self._internal_events = []

    def smdp_discount(self, gamma: float, elapsed_ticks: int) -> float:
        return gamma ** elapsed_ticks
