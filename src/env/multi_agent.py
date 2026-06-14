"""Two-agent game loop for the first SACRED test environment.

This is intentionally not a PettingZoo wrapper yet. It keeps the first
adversarial loop easy to inspect while preserving the core API shape needed
for a later wrapper: protagonist action, antagonist action, environment step,
and per-agent rewards.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import hypot
from typing import Callable, Protocol

import networkx as nx

from src.env.graph_env import EdgeId, GraphEnv, NodeId, StepResult
from src.env.toy_graph import make_toy_graph_env


@dataclass(frozen=True, slots=True)
class GameConfig:
    """Configuration for the deterministic toy adversarial game."""

    tick_seconds: float = 1.0
    max_ticks: int = 240
    congestion_level: float = 0.65
    congestion_duration: int = 12
    congestion_budget: float = 72.0
    congestion_cooldown: int = 3
    time_penalty: float = 0.04
    remaining_demand_penalty: float = 0.08
    congestion_cost: float = 0.015


@dataclass(slots=True)
class CongestionBudget:
    """Rolling accounting for the antagonist's limited disruption capacity."""

    total: float
    used: float = 0.0

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.used)

    def can_spend(self, amount: float) -> bool:
        return amount <= self.remaining + 1e-12

    def spend(self, amount: float) -> bool:
        if not self.can_spend(amount):
            return False
        self.used += amount
        return True


@dataclass(slots=True)
class EpisodeMetrics:
    """Episode-level metrics used by tests, renderer overlays, and later logs."""

    ticks: int = 0
    total_delivery: float = 0.0
    total_distance: float = 0.0
    protagonist_return: float = 0.0
    antagonist_return: float = 0.0
    congestion_budget_used: float = 0.0
    congestion_events: int = 0
    done_reason: str = "running"


@dataclass(slots=True)
class GameTick:
    """Single tick record returned by :class:`SacredToyGame`."""

    step_result: StepResult
    protagonist_reward: float
    antagonist_reward: float
    protagonist_action: dict[int, NodeId]
    antagonist_action: dict[EdgeId, float]
    metrics: EpisodeMetrics


class ProtagonistPolicy(Protocol):
    def act(self, env: GraphEnv) -> dict[int, NodeId]:
        """Return dispatch actions for idle trucks."""


class AntagonistPolicy(Protocol):
    def act(self, env: GraphEnv, game: SacredToyGame) -> dict[EdgeId, float]:
        """Return congestion actions for this tick."""


class NearestDemandProtagonist:
    """Baseline dispatcher with depot reload cycles."""

    def act(self, env: GraphEnv) -> dict[int, NodeId]:
        actions = {}
        
        # Calculate expected demand (actual minus what inbound trucks will deliver)
        expected_demand = {
            node: data["demand"] for node, data in env.graph.nodes(data=True) if data["demand"] > 0
        }
        for truck in env.trucks.values():
            if truck.destination is not None and truck.destination in expected_demand:
                expected_demand[truck.destination] -= truck.load

        demand_nodes = {
            node for node, rem in expected_demand.items() 
            if rem > 0 and not env.graph.nodes[node]["has_depot"]
        }

        for truck_id, truck in env.trucks.items():
            if not truck.is_idle:
                continue

            if truck.load <= 0 and truck.current_node != truck.home_depot:
                actions[truck_id] = truck.home_depot
                continue

            if truck.load <= 0:
                continue

            truck_comp = env.node_to_component.get(truck.current_node)
            candidates = sorted([n for n in demand_nodes if env.node_to_component.get(n) == truck_comp])

            if not candidates and truck.current_node != truck.home_depot:
                if env.node_to_component.get(truck.home_depot) == truck_comp:
                    actions[truck_id] = truck.home_depot
                continue
                
            if not candidates:
                continue

            try:
                distances = nx.single_source_dijkstra_path_length(env.graph, truck.current_node, weight="distance")
                destination = min(candidates, key=lambda node: distances.get(node, float("inf")))
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                destination = candidates[0]
                
            actions[truck_id] = destination
            
            # Immediately update expected demand for the next truck in this loop
            expected_demand[destination] -= truck.load
            if expected_demand[destination] <= 0:
                demand_nodes.remove(destination)

        return actions


class RouteInterceptingAntagonist:
    """Baseline adversary: congest the active route edge with largest delay."""

    def act(self, env: GraphEnv, game: SacredToyGame) -> dict[EdgeId, float]:
        if game.cooldown_remaining > 0:
            return {}
        if not game.budget.can_spend(game.congestion_action_cost):
            return {}

        candidates = self._active_route_edges(env)
        if not candidates:
            candidates = sorted(list(env.graph.edges))
        if not candidates:
            return {}

        edge = max(candidates, key=lambda item: self._edge_score(env, item))
        return {edge: game.config.congestion_level}

    def _active_route_edges(self, env: GraphEnv) -> list[EdgeId]:
        edges: list[EdgeId] = []
        for truck in env.trucks.values():
            if truck.edge is not None:
                edges.append(truck.edge)
                continue
            if truck.destination is None or truck.path_index >= len(truck.path) - 1:
                continue
            edges.append((truck.path[truck.path_index], truck.path[truck.path_index + 1]))
        return edges

    def _edge_score(self, env: GraphEnv, edge: EdgeId) -> float:
        u, v = edge
        distance = env.graph.edges[u, v]["distance"]
        demand_gravity = self._downstream_demand_gravity(env, u, v)
        return distance + demand_gravity

    def _downstream_demand_gravity(self, env: GraphEnv, u: NodeId, v: NodeId) -> float:
        ux, uy = env.graph.nodes[u]["x"], env.graph.nodes[u]["y"]
        vx, vy = env.graph.nodes[v]["x"], env.graph.nodes[v]["y"]
        score = 0.0
        for _, data in env.graph.nodes(data=True):
            demand = data["demand"]
            if demand <= 0:
                continue
            midpoint_distance = hypot(((ux + vx) / 2.0) - data["x"], ((uy + vy) / 2.0) - data["y"])
            score += demand / max(1.0, midpoint_distance)
        return score


class NoOpAntagonist:
    """Baseline clean-network opponent."""

    def act(self, env: GraphEnv, game: SacredToyGame) -> dict[EdgeId, float]:
        return {}


class SacredToyGame:
    """Deterministic two-agent testbed for early SACRED experiments."""

    def __init__(
        self,
        *,
        env_factory: Callable[[], GraphEnv] = make_toy_graph_env,
        protagonist: ProtagonistPolicy | None = None,
        antagonist: AntagonistPolicy | None = None,
        config: GameConfig | None = None,
    ) -> None:
        self.env_factory = env_factory
        self.protagonist = protagonist or NearestDemandProtagonist()
        self.antagonist = antagonist or RouteInterceptingAntagonist()
        self.config = config or GameConfig()
        self.env = self.env_factory()
        self.budget = CongestionBudget(self.config.congestion_budget)
        self.active_congestion: dict[EdgeId, int] = {}
        self.cooldown_remaining = 0
        self.metrics = EpisodeMetrics()

    @property
    def congestion_action_cost(self) -> float:
        return self.config.congestion_level * self.config.congestion_duration

    def reset(self) -> dict:
        self.env = self.env_factory()
        self.env.max_time = self.config.max_ticks
        self.env.reset()
        self.budget = CongestionBudget(self.config.congestion_budget)
        self.active_congestion = {}
        self.cooldown_remaining = 0
        self.metrics = EpisodeMetrics()
        return self.env.observe()

    def step(self) -> GameTick:
        self._age_congestion()
        antagonist_action = self.antagonist.act(self.env, self)
        accepted_antagonist_action = self._accept_antagonist_action(antagonist_action)
        protagonist_action = self.protagonist.act(self.env)

        result = self.env.step(
            dispatch_actions=protagonist_action,
            congestion_actions=accepted_antagonist_action,
        )
        protagonist_reward, antagonist_reward = self._agent_rewards(result, accepted_antagonist_action)
        self._update_metrics(result, protagonist_reward, antagonist_reward, accepted_antagonist_action)
        return GameTick(
            step_result=result,
            protagonist_reward=protagonist_reward,
            antagonist_reward=antagonist_reward,
            protagonist_action=protagonist_action,
            antagonist_action=accepted_antagonist_action,
            metrics=self._metrics_snapshot(),
        )

    def run_episode(self) -> EpisodeMetrics:
        self.reset()
        while not self.env.is_done() and self.metrics.ticks < self.config.max_ticks:
            self.step()
        if self.metrics.done_reason == "running":
            self.metrics.done_reason = "max_ticks"
        return self._metrics_snapshot()

    def _accept_antagonist_action(self, action: dict[EdgeId, float]) -> dict[EdgeId, float]:
        accepted: dict[EdgeId, float] = {}
        for edge, level in action.items():
            if edge in self.active_congestion:
                continue
                
            # Snap level to nearest discrete bin
            bins = [0.0, 0.25, 0.5, 0.75, 1.0]
            snapped_level = min(bins, key=lambda b: abs(b - float(level)))
            
            if snapped_level == 0.0:
                continue
                
            cost = snapped_level * self.config.congestion_duration
            if not self.budget.spend(cost):
                continue
                
            accepted[edge] = snapped_level
            self.active_congestion[edge] = self.config.congestion_duration
            self.cooldown_remaining = self.config.congestion_cooldown
            break
        return accepted

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

    def _agent_rewards(self, result: StepResult, antagonist_action: dict[EdgeId, float]) -> tuple[float, float]:
        delivered = sum(delivery["delivered"] for delivery in result.info["deliveries"])
        remaining_demand = self._remaining_demand()
        congestion_spend = sum(antagonist_action.values()) * self.config.congestion_duration
        protagonist_reward = (
            delivered
            - self.config.time_penalty
            - (self.config.remaining_demand_penalty * remaining_demand)
            - (self.config.congestion_cost * congestion_spend)
        )
        antagonist_reward = -protagonist_reward - (self.config.congestion_cost * congestion_spend)
        return protagonist_reward, antagonist_reward

    def _update_metrics(
        self,
        result: StepResult,
        protagonist_reward: float,
        antagonist_reward: float,
        antagonist_action: dict[EdgeId, float],
    ) -> None:
        self.metrics.ticks = result.observation["time"]
        self.metrics.total_delivery += sum(delivery["delivered"] for delivery in result.info["deliveries"])
        self.metrics.total_distance += result.info["distance_travelled"]
        self.metrics.protagonist_return += protagonist_reward
        self.metrics.antagonist_return += antagonist_reward
        self.metrics.congestion_budget_used = self.budget.used
        self.metrics.congestion_events += len(antagonist_action)
        if self._remaining_demand() <= 0 and self._all_trucks_home():
            self.metrics.done_reason = "served_all_demand"
        elif self.metrics.ticks >= self.config.max_ticks:
            self.metrics.done_reason = "max_ticks"

    def _remaining_demand(self) -> float:
        return sum(data["demand"] for _, data in self.env.graph.nodes(data=True))

    def _all_trucks_home(self) -> bool:
        return all(
            truck.current_node == self.env.depot_node and truck.edge is None for truck in self.env.trucks.values()
        )

    def _metrics_snapshot(self) -> EpisodeMetrics:
        return EpisodeMetrics(
            ticks=self.metrics.ticks,
            total_delivery=self.metrics.total_delivery,
            total_distance=self.metrics.total_distance,
            protagonist_return=self.metrics.protagonist_return,
            antagonist_return=self.metrics.antagonist_return,
            congestion_budget_used=self.metrics.congestion_budget_used,
            congestion_events=self.metrics.congestion_events,
            done_reason=self.metrics.done_reason,
        )


def describe_game_tick(tick: GameTick) -> list[tuple[str, str]]:
    """Return compact human-readable protagonist/antagonist activity messages."""

    messages: list[tuple[str, str]] = []
    time = tick.metrics.ticks
    depot_nodes = {
        node for node, data in tick.step_result.observation["nodes"].items() if data["has_depot"]
    }

    for truck_id, destination in tick.protagonist_action.items():
        if destination == tick.step_result.observation["trucks"][truck_id]["current_node"]:
            messages.append(("P", f"t={time}: truck {truck_id} serving customer {destination}"))
        elif destination in depot_nodes:
            messages.append(("P", f"t={time}: truck {truck_id} returning to depot"))
        else:
            messages.append(("P", f"t={time}: truck {truck_id} fulfilling customer {destination}"))

    for delivery in tick.step_result.info["deliveries"]:
        messages.append(
            (
                "P",
                f"t={time}: truck {delivery['truck_id']} delivered {delivery['delivered']:.0f} at {delivery['node']}",
            )
        )

    for reload_event in tick.step_result.info["reloads"]:
        messages.append(
            (
                "P",
                f"t={time}: truck {reload_event['truck_id']} reloaded {reload_event['reloaded']:.0f} at depot",
            )
        )

    for edge, level in tick.antagonist_action.items():
        messages.append(("A", f"t={time}: blocking edge {edge[0]}-{edge[1]} at {level:.0%}"))

    return messages
