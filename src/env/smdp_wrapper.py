"""SMDP decision wrapper for SACRED training.

The low-level :class:`GraphEnv` ticks every simulated second. This wrapper
skips non-decision ticks and yields only meaningful decision events for the
protagonist or antagonist, while accumulating rewards and elapsed time for
SMDP replay records.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    ) -> None:
        self.env_factory = env_factory
        self.config = config or SMDPConfig()
        self.env = self.env_factory()
        self.budget = CongestionBudget(self.config.congestion_budget)
        self.active_congestion: dict[EdgeId, int] = {}
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

    def reset_decision_env(self) -> DecisionEvent:
        """Reset and return the first decision event."""

        self.env = self.env_factory()
        self.env.max_time = self.config.max_ticks
        self.env.reset()
        self.budget = CongestionBudget(self.config.congestion_budget)
        self.active_congestion = {}
        self.cooldown_remaining = 0
        self.next_antagonist_tick = self.config.antagonist_interval
        self._last_decision_observation = None
        self._last_action = None
        self._last_action_mask = {}
        self._last_agent = None
        self._reset_accumulators()
        self._in_sequential_epoch = False
        return self._build_event(DecisionType.PROTAGONIST_DECISION)

    def step_protagonist(self, actions_by_truck: Mapping[int, NodeId]) -> tuple[DecisionEvent, SMDPTransition]:
        """Apply protagonist decisions and advance to the next event."""

        state = self.env.observe()
        action_mask = self.protagonist_action_mask()
        dispatch_actions = self._valid_protagonist_actions(actions_by_truck, action_mask)
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
            
            # GPS-style dynamic rerouting for trucks affected by this newly injected congestion
            if level > 0.0:
                for truck_id, truck in self.env.trucks.items():
                    if truck.is_idle or not truck.path:
                        continue
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
                        if truncate_idx <= len(truck.path):
                            next_intersection = truck.path[truncate_idx - 1]
                            print(f"[REROUTE] Antagonist congested edge {edge} (level {level}). Truncating Truck {truck_id} path from destination {truck.destination} to next intersection {next_intersection}.")
                            truck.path = list(truck.path[:truncate_idx])
                            truck.destination = truck.path[-1]

        # Determine if we continue the sequential decision loop
        min_cost = min(self.config.congestion_levels) * self.config.congestion_duration
        
        # Temporarily enable sequential epoch flag to generate mask ignoring cooldown
        original_seq_flag = getattr(self, "_in_sequential_epoch", False)
        self._in_sequential_epoch = True
        next_mask = self.antagonist_action_mask()
        has_allowed_edges = any(next_mask.get("levels_by_edge", {}).values())
        self._in_sequential_epoch = original_seq_flag

        continue_loop = (
            action is not None
            and applied_any
            and self.budget.remaining >= min_cost - 1e-6
            and has_allowed_edges
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
                reward=0.0,
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

        immediate = self._current_decision_type()
        if immediate is not None:
            return self._build_event(immediate)

        while True:
            step_result = self.env.step()
            self._accumulate_step(step_result, antagonist_action={})
            self._age_congestion()

            decision_type = self._current_decision_type()
            if decision_type is not None:
                return self._build_event(decision_type)

    def protagonist_action_mask(self) -> dict[int, list[NodeId]]:
        """Return valid next-node actions for all waiting trucks."""

        mask: dict[int, list[NodeId]] = {}
        
        # Precompute total targeted loads for each destination
        targeted_loads: dict[NodeId, float] = {}
        for t in self.env.trucks.values():
            if t.destination is not None:
                targeted_loads[t.destination] = targeted_loads.get(t.destination, 0.0) + t.load

        for truck_id, truck in self.env.trucks.items():
            if not truck.is_idle:
                continue

            current_node = truck.current_node
            if current_node is None:
                continue

            if truck.load <= 0 and current_node != truck.home_depot:
                mask[truck_id] = [truck.home_depot]
            elif truck.load <= 0:
                mask[truck_id] = []
            elif self._remaining_demand() <= 0 and current_node != truck.home_depot:
                mask[truck_id] = [truck.home_depot]
            elif self._remaining_demand() <= 0:
                mask[truck_id] = []
            else:
                # Goal-Directed: All customer nodes with positive unassigned demand, plus the depot
                truck_comp = self.env.node_to_component.get(current_node)
                destinations = []
                for n, data in self.env.graph.nodes(data=True):
                    if self.env.node_to_component.get(n) != truck_comp:
                        continue
                    if not data.get("has_depot", False):
                        node_demand = float(data.get("demand", 0.0))
                        if node_demand > 0.0:
                            # Subtract in-transit/committed load of all OTHER trucks targeting this node
                            other_targeted = targeted_loads.get(n, 0.0)
                            if truck.destination == n:
                                other_targeted -= truck.load
                            unassigned_demand = node_demand - other_targeted
                            if unassigned_demand > 0.0:
                                destinations.append(n)
                
                # Only allow depot if load < capacity, OR if the truck is NOT already at the depot!
                if truck.load < truck.capacity or current_node != truck.home_depot:
                    if self.env.node_to_component.get(truck.home_depot) == truck_comp:
                        destinations.append(truck.home_depot)
                
                # Filter out the current node (no choosing to stay at current customer)
                if current_node in destinations and current_node != truck.home_depot:
                    destinations.remove(current_node)
                mask[truck_id] = destinations
        return mask

    def antagonist_action_mask(self) -> dict[str, Any]:
        """Return valid antagonist action choices at the current event with 3-road Action Masking."""

        if not getattr(self, "_in_sequential_epoch", False) and self.cooldown_remaining > 0:
            return {"can_wait": True, "levels_by_edge": {}}

        # 1. Action-Space Masking (O(1) 3-road radius from trucks)
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
        for edge in nearby_edges:
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

        protagonist_due = any(options for options in self.protagonist_action_mask().values())
        antagonist_due = self.env.time >= self.next_antagonist_tick

        if protagonist_due and antagonist_due:
            return DecisionType.BOTH_DECISION
        if antagonist_due:
            return DecisionType.ANTAGONIST_DECISION
        if protagonist_due:
            return DecisionType.PROTAGONIST_DECISION
        return None

    def _build_event(self, decision_type: DecisionType) -> DecisionEvent:
        event = DecisionEvent(
            decision_type=decision_type,
            observation=self.env.observe(),
            waiting_trucks=[truck_id for truck_id, options in self.protagonist_action_mask().items() if options],
            protagonist_reward=self._accumulated_protagonist_reward,
            antagonist_reward=self._accumulated_antagonist_reward,
            elapsed_ticks=self._elapsed_ticks,
            protagonist_action_mask=self.protagonist_action_mask(),
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

        self.active_congestion[edge] = self.config.congestion_duration
        self.cooldown_remaining = self.config.congestion_cooldown
        return {edge: level}

    def _accumulate_step(self, step_result: StepResult, antagonist_action: Mapping[EdgeId, float]) -> None:
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
        expired: list[EdgeId] = []
        for edge, ticks_remaining in self.active_congestion.items():
            next_ticks = ticks_remaining - 1
            if next_ticks <= 0:
                expired.append(edge)
            else:
                self.active_congestion[edge] = next_ticks

        for edge in expired:
            self.active_congestion.pop(edge, None)
            if self.env.graph.has_edge(*edge):
                self.env.set_congestion(edge, 0.0)

        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1

    def _remaining_demand(self) -> float:
        return self.env.remaining_demand

    def _neighbors_toward(self, source: NodeId, target: NodeId) -> list[NodeId]:
        try:
            path = nx.shortest_path(self.env.graph, source, target, weight="distance")
        except nx.NetworkXNoPath:
            return []
        if len(path) < 2:
            return []
        return [path[1]]

    def _reset_accumulators(self) -> None:
        self._accumulated_protagonist_reward = 0.0
        self._accumulated_antagonist_reward = 0.0
        self._elapsed_ticks = 0
        self._internal_events = []
