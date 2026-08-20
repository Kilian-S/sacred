"""SMDP decision wrapper for SACRED training, sitting on the one-second ticks of
:class:`GraphEnv`. It skips non-decision ticks, yields only meaningful decision events for the
protagonist or antagonist, and accumulates rewards and elapsed time for SMDP replay records.
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
    # Reward shaping selector. "legacy" uses the delivery, time and remaining-demand shaping
    # above. "latency" pays -1 per tick per outstanding arrived-but-undelivered request, which
    # telescopes over the episode to the total delivery latency.
    reward_mode: str = "legacy"
    # Protagonist action model. "destination" picks a target node and the env routes there.
    # "next_hop" picks the next adjacent node one edge at a time, so the policy owns the route and
    # the antagonist's congestion becomes an exploitable decision. See dispatch_truck_edge.
    routing_mode: str = "destination"
    # Next-hop only: a hop is allowed if the route through it stays within this factor of the
    # shortest static distance to the goal. Above 1.0 this keeps near-shortest alternatives, the
    # safe route among them, while excluding detours that would let the truck wander the map.
    routing_corridor_slack: float = 1.5
    # Antagonist sequential-epoch cap: the most congestion sub-actions allowed per decision event,
    # 0 meaning unlimited. Each sub-action is a stored transition and, in the antagonist phase, a
    # gradient update, so an uncapped epoch makes the antagonist phase far slower than the
    # protagonist. A cap of 1 places one sustained roadblock per event and keeps the two agents'
    # update counts comparable.
    max_antag_actions_per_event: int = 0
    # Antagonist reach, meaning which edges it may block. "leashed" is the 3-hop radius around
    # trucks, so it blocks reactively right in front of them. "route" is the edges on each truck's
    # shortest path to its target, letting it pre-block the gateway ahead and so giving the
    # protagonist something to anticipate.
    antag_reach: str = "leashed"
    # Reward baseline. "none" leaves the raw latency reward. "twin" subtracts a per-tick,
    # action-independent baseline b(t), so protagonist_reward = -(remaining_demand - b(t)), with
    # b(t) supplied by a baseline_provider injected into the wrapper. Because b(t) depends on
    # neither agent's actions, the episode reward shifts by the constant sum_t b(t): the zero-sum
    # game and its equilibrium are preserved and only the gradient variance changes, stripping the
    # arrival trend and the damage that is unavoidable even under clean play.
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
    # Cache of featurised graphs, keyed "state" and "next", filled on the first update that
    # samples this transition. Featurisation is a pure function of the buffered state, so caching
    # it preserves behaviour, and the cache is freed with the transition on eviction.
    feature_cache: dict[str, Any] = field(default_factory=dict)

    def __getstate__(self) -> dict[str, Any]:
        # Keep the feature cache, which can run to gigabytes, out of the pickle so that a
        # checkpointed replay buffer stays small. The live cache is untouched and simply rebuilds
        # itself lazily after a resume.
        return {f.name: getattr(self, f.name) for f in fields(self) if f.name != "feature_cache"}

    def __setstate__(self, state: Any) -> None:
        # `state` is either a plain dict or a `(None, slot_dict)` tuple, the form older pickles of
        # this slotted dataclass take. Handle both.
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
        # Reward baseline, see SMDPConfig.reward_baseline. `baseline_provider(env)` returns a
        # `(series, last)` pair mapping post-step env.time to b(t), plus a pad value for ticks past
        # the provider's range, and is called once per reset. `_baseline_record` is the recording
        # buffer that captures a series during a provider's own rollout; it is set externally,
        # never by reset, so the one mechanism both records and subtracts.
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
        # Compute this episode's action-independent baseline from the freshly generated arrivals,
        # which the provider replays clean. Skipped when disabled or when this wrapper is itself a
        # provider's twin. `_baseline_record` is left untouched so that a provider can capture a
        # series across this reset.
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
            # Split by decision type: an unassigned truck's action is an assignment, which sets the
            # target without moving; an assigned truck's action is a next-hop of one edge.
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
            
            # GPS-style dynamic rerouting for trucks affected by this newly injected congestion,
            # in destination mode only. Under next-hop routing the policy owns rerouting, so
            # doing it here would erase the very decision the protagonist must learn.
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

        min_cost = min(self.config.congestion_levels) * self.config.congestion_duration

        # Force the sequential-epoch flag on so the mask is built ignoring the cooldown.
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

        cost_penalty = self.config.congestion_cost * congestion_spend
        self._accumulated_antagonist_reward -= cost_penalty

        if continue_loop:
            # Stay inside the sequential epoch: no simulated time passes, so elapsed_ticks is 0.
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
            # End the sequential epoch and advance simulated time to the next decision event.
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
                # Every customer node with positive unassigned demand, plus the depot.
                destinations = []
                for n, node_demand in valid_customers_by_comp.get(truck_comp, {}).items():
                    # Net off the committed load of every other truck already targeting this node.
                    other_targeted = targeted_loads.get(n, 0.0)
                    if truck.destination == n:
                        other_targeted -= truck.load
                    unassigned_demand = node_demand - other_targeted
                    if unassigned_demand > 0.0:
                        destinations.append(n)
                
                # The depot is only a valid target for a truck with room to reload or one that is
                # not already standing there.
                if truck.load < truck.capacity or current_node != truck.home_depot:
                    if can_reach_depot:
                        destinations.append(truck.home_depot)

                # A truck may not choose to stay put at the customer it is standing on.
                if current_node in destinations and current_node != truck.home_depot:
                    destinations.remove(current_node)
                mask[truck_id] = destinations
        return mask

    def _next_hop_action_mask(self) -> dict[int, list[NodeId]]:
        """Forward next-hop neighbours for every idle truck, by static congestion-free distance.

        A forward neighbour is one that makes progress toward the truck's goal, the nearest demand
        if it is loaded and otherwise the home depot. This bounds exploration to a corridor while
        preserving the route choice, since at a branch both the fast and the safe route are
        forward and the policy decides between them. Congestion is read from the observation
        rather than the mask, so the antagonist can still make the fast route costly.
        """
        mask: dict[int, list[NodeId]] = {}
        for truck_id, truck in self.env.trucks.items():
            if not truck.is_idle or truck.current_node is None:
                continue
            mask[truck_id] = self._forward_mask(truck, self._truck_goal(truck))
        return mask

    def _forward_mask(self, truck: Any, goal: NodeId | None) -> list[NodeId]:
        """Corridor mask of forward next-hop neighbours toward ``goal``, with anti-oscillation.

        Shared by next-hop and hybrid routing, the latter passing the truck's ``assigned_target``.
        """
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
        # Anti-oscillation: drop the node just left only when going back there would be a backward
        # move, which still allows a legitimate turnaround when the goal flips to the depot.
        if truck.path_index >= 1 and len(truck.path) > truck.path_index:
            prev = truck.path[truck.path_index - 1]
            if dist_to_goal.get(prev, float("inf")) >= cur:
                pruned = [n for n in forward if n != prev]
                if pruned:
                    forward = pruned
        return forward

    def _hybrid_action_mask(self) -> dict[int, list[NodeId]]:
        """Hybrid mask, where each idle truck's decision type follows from its state.

        A loaded truck with no assigned target faces an assignment and picks a pending request; a
        truck that already has a target, whether a request or the depot on its return leg, faces a
        routing choice among forward next-hops. The env manages ``assigned_target`` itself.
        """
        mask: dict[int, list[NodeId]] = {}
        for truck_id, truck in self.env.trucks.items():
            if not truck.is_idle or truck.current_node is None:
                continue
            if truck.assigned_target is not None:
                mask[truck_id] = self._forward_mask(truck, truck.assigned_target)
            elif truck.load > 0:
                mask[truck_id] = self._assignment_candidates(truck)
            else:  # empty and unassigned, so route home to reload
                mask[truck_id] = self._forward_mask(truck, truck.home_depot)
        return mask

    def _assignment_candidates(self, truck: Any) -> list[NodeId]:
        """Pending request nodes in the truck's own component, minus those another truck already
        holds.

        The exclusion is the cross-event half of claiming, the same-event half being applied by the
        trainer and the evaluator; without it a request could be double-assigned and the second
        truck stranded at a node the first had already emptied. With no unclaimed request left the
        truck is sent home so the episode can terminate, and since the env clears the assignment on
        arrival it re-enters assignment if new work appears.
        """
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
        """Dispatch idle trucks that have exactly one forward routing option, so that a protagonist
        decision surfaces only at a genuine branch. Assignments are never auto-resolved."""
        mode = self.config.routing_mode
        if mode == "next_hop":
            for truck_id, options in self._next_hop_action_mask().items():
                if len(options) == 1:
                    self.env.dispatch_truck_edge(truck_id, options[0])
        elif mode == "hybrid":
            for truck_id, truck in self.env.trucks.items():
                if truck.assigned_target is None or not truck.is_idle or truck.current_node is None:
                    continue  # only already-assigned trucks auto-resolve
                options = self._forward_mask(truck, truck.assigned_target)
                if len(options) == 1:
                    self.env.dispatch_truck_edge(truck_id, options[0])

    def _route_reach_edges(self) -> set:
        """Edges on each truck's static shortest path to its target, the adversary's route reach.

        Using congestion-free distance gives the truck's intended route, so a block placed on it
        forces the truck to detour rather than the adversary chasing a reroute. A truck with no
        target, waiting at a depot for an assignment, is not yet committed and contributes nothing.
        """
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
        """Return the antagonist's valid action choices at the current event."""

        if not getattr(self, "_in_sequential_epoch", False) and self.cooldown_remaining > 0:
            return {"can_wait": True, "levels_by_edge": {}}

        # Which edges the antagonist may block, that is its reach.
        if self.config.antag_reach == "route":
            nearby_edges = self._route_reach_edges()
        else:
            # Leashed reach: the 3-hop radius around trucks, from the precomputed k-hop sets.
            nearby_edges = set()
            for truck in self.env.trucks.values():
                if truck.current_node is not None:
                    nearby_edges.update(self.env._k_hop_edges[truck.current_node])
                elif truck.edge is not None:
                    u, v = truck.edge
                    nearby_edges.update(self.env._k_hop_edges[u])
                    nearby_edges.update(self.env._k_hop_edges[v])

        # Then keep only the levels the remaining budget can still pay for.
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
            # Only a branch is a real decision; forced single-option moves are auto-resolved.
            protagonist_due = any(len(options) >= 2 for options in mask.values())
        elif self.config.routing_mode == "hybrid":
            # An assignment is due as soon as one candidate exists, a routing move only at a
            # branch, since single-option routing moves are auto-resolved.
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
            # Delivery-latency objective: pay -1 per tick per outstanding demand unit, counted
            # after this tick's deliveries, so completing a unit immediately stops accruing its
            # penalty. Over an episode this telescopes to the total latency. Counting units rather
            # than nodes keeps it correct when one node holds many units.
            remaining = float(self.env.remaining_demand)
            # Recording pass: capture this tick's remaining_demand for a provider's twin series.
            if self._baseline_record is not None:
                self._baseline_record.append((int(self.env.time), remaining))
            # Subtraction pass: strip the action-independent baseline b(t), padding ticks past the
            # provider's range with the twin's final value, since a real episode under attack runs
            # longer than its clean twin. b(t) is constant in both agents' actions, so the game
            # stays zero-sum and the telescoped total shifts by a per-episode constant.
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

        Where all requests are present from t=0 this is just the number of non-depot nodes with
        positive remaining demand. The env keeps that set in ``valid_customers_by_comp``; if it is
        absent, fall back to a graph scan.
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
