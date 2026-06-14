"""Headless graph simulation for the SACRED SDVRP environment.

The PettingZoo wrapper and renderer should sit on top of this module. This
file owns only the fast tick-by-tick state transitions: graph attributes,
truck dispatching, routing, congestion updates, and movement physics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
from math import hypot
from typing import Any, Hashable, Iterable, Mapping

import networkx as nx
import numpy as np


NodeId = Hashable
EdgeId = tuple[NodeId, NodeId]


@dataclass(slots=True)
class TruckState:
    """Mutable state for one truck in the headless simulator."""

    truck_id: int
    speed: float
    current_node: NodeId | None
    home_depot: NodeId | None = None
    destination: NodeId | None = None
    path: list[NodeId] = field(default_factory=list)
    path_index: int = 0
    edge: EdgeId | None = None
    edge_progress: float = 0.0
    capacity: float = 1.0
    load: float = 1.0
    delivered_total: float = 0.0

    @property
    def is_idle(self) -> bool:
        return self.edge is None and self.current_node is not None


@dataclass(frozen=True, slots=True)
class StepResult:
    """Structured result returned by :meth:`GraphEnv.step`."""

    observation: dict[str, Any]
    reward: float
    done: bool
    info: dict[str, Any]


class GraphEnv:
    """Core SDVRP graph environment without PettingZoo or PyGame concerns.

    Parameters
    ----------
    graph:
        Optional NetworkX graph. Nodes should expose ``x``, ``y``, ``demand``,
        and ``has_depot`` attributes. Edges should expose ``distance`` and
        ``congestion_level`` attributes. Missing attributes are filled with
        defaults.
    num_trucks:
        Number of trucks initialized at the depot.
    truck_speed:
        Base distance a truck can travel in one tick when congestion is zero.
    truck_capacity:
        Amount of demand fulfilled when a truck reaches a customer node.
    depot_node:
        Explicit depot node. If omitted, the first node with ``has_depot=True``
        is used.
    """

    def __init__(
        self,
        graph: nx.Graph | None = None,
        *,
        nodes: Mapping[NodeId, Mapping[str, Any]] | None = None,
        edges: Iterable[tuple[NodeId, NodeId, Mapping[str, Any]]] | None = None,
        num_trucks: int = 1,
        truck_speed: float = 1.0,
        truck_capacity: float = 1.0,
        depot_node: NodeId | None = None,
        truck_starting_nodes: list[NodeId] | None = None,
        max_time: int | None = None,
    ) -> None:
        if num_trucks < 1:
            raise ValueError("num_trucks must be at least 1")
        if truck_speed <= 0:
            raise ValueError("truck_speed must be positive")
        if truck_capacity <= 0:
            raise ValueError("truck_capacity must be positive")

        self.graph = self._build_graph(graph, nodes, edges)
        resolved_depot = self._resolve_depot(depot_node)
        self.depot_node = resolved_depot
        
        if truck_starting_nodes is not None:
            self.truck_starting_nodes = truck_starting_nodes
            self.num_trucks = len(truck_starting_nodes)
        else:
            self.truck_starting_nodes = [resolved_depot] * num_trucks
            self.num_trucks = num_trucks

        # Ensure depots never have demand
        for node, data in self.graph.nodes(data=True):
            if data.get("has_depot", False):
                data["demand"] = 0.0
                
        self._initial_graph = self.graph.copy()
        self.truck_speed = float(truck_speed)
        self.truck_capacity = float(truck_capacity)
        self.max_time = max_time

        self.time = 0
        self.trucks: dict[int, TruckState] = {}
        
        # Precompute connected components to prevent NetworkXNoPath crashes
        self.node_to_component: dict[NodeId, int] = {}
        for i, comp in enumerate(nx.connected_components(self.graph)):
            for node in comp:
                self.node_to_component[node] = i
                
        import collections
        # Precompute 3-hop reachable edges for O(1) antagonist masking
        self._k_hop_edges: dict[NodeId, set[tuple[NodeId, NodeId]]] = {}
        for node in self.graph.nodes():
            nearby = set()
            queue = collections.deque([(node, 0)])
            visited = {node}
            while queue:
                curr, depth = queue.popleft()
                if depth >= 3:
                    continue
                try:
                    neighbors = list(self.graph.successors(curr))
                except AttributeError:
                    neighbors = list(self.graph.neighbors(curr))
                for neighbor in neighbors:
                    if self.graph.has_edge(curr, neighbor):
                        nearby.add(self._edge_key(curr, neighbor))
                    if self.graph.has_edge(neighbor, curr):
                        nearby.add(self._edge_key(neighbor, curr))
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append((neighbor, depth + 1))
            self._k_hop_edges[node] = nearby
            
        self.reset()

    def reset(self) -> dict[str, Any]:
        """Reset the simulation clock and return the initial observation."""

        self.graph = self._initial_graph.copy()
        self.time = 0
        self.trucks = {
            truck_id: TruckState(
                truck_id=truck_id,
                speed=self.truck_speed,
                current_node=self.truck_starting_nodes[truck_id],
                home_depot=self.truck_starting_nodes[truck_id],
                capacity=self.truck_capacity,
                load=self.truck_capacity,
                path=[self.truck_starting_nodes[truck_id]],
            )
            for truck_id in range(self.num_trucks)
        }
        
        # Track global demand for O(1) is_done() checks
        self.remaining_demand = sum(data["demand"] for _, data in self.graph.nodes(data=True))
        
        self._obs_nodes = {
            node: {
                "x": data["x"],
                "y": data["y"],
                "demand": data["demand"],
                "has_depot": data["has_depot"],
            }
            for node, data in self.graph.nodes(data=True)
        }
        
        self._obs_edges = {
            self._edge_key(u, v): {
                "distance": data["distance"],
                "congestion_level": data["congestion_level"],
            }
            for u, v, data in self.graph.edges(data=True)
        }
        
        return self.observe()

    def step(
        self,
        dispatch_actions: Mapping[int, NodeId] | None = None,
        congestion_actions: Mapping[EdgeId, float] | Iterable[tuple[NodeId, NodeId, float]] | None = None,
    ) -> StepResult:
        """Advance the environment by one tick.

        ``dispatch_actions`` maps idle truck ids to destination node ids.
        ``congestion_actions`` sets edge congestion levels before movement.
        A congestion level of ``0.0`` means free flow, ``0.5`` means half speed,
        and ``1.0`` means blocked.
        """

        info: dict[str, Any] = {
            "time": self.time + 1,
            "dispatched": [],
            "ignored_dispatches": [],
            "congestion_updates": [],
            "arrivals": [],
            "deliveries": [],
            "reloads": [],
            "distance_travelled": 0.0,
        }

        self._apply_congestion(congestion_actions, info)
        self._apply_dispatch(dispatch_actions, info)

        for truck in self.trucks.values():
            info["distance_travelled"] += self._move_truck_one_tick(truck, info)

        self.time += 1
        reward = self._reward(info)
        done = self.is_done()
        return StepResult(self.observe(), reward, done, info)

    def observe(self) -> dict[str, Any]:
        """Return a Python-dict observation suitable for wrappers to transform."""
        return {
            "time": self.time,
            "nodes": self._obs_nodes,
            "edges": self._obs_edges,
            "trucks": {
                truck_id: {
                    "current_node": truck.current_node,
                    "destination": truck.destination,
                    "edge": truck.edge,
                    "edge_progress": truck.edge_progress,
                    "capacity": truck.capacity,
                    "load": truck.load,
                    "path": list(truck.path),
                    "path_index": truck.path_index,
                    "delivered_total": truck.delivered_total,
                }
                for truck_id, truck in self.trucks.items()
            },
        }



    def is_done(self) -> bool:
        """Return true when all demand is served or the optional horizon ends."""

        if self.max_time is not None and self.time >= self.max_time:
            return True
            
        if self.remaining_demand > 0:
            return False
            
        # O(1) check if all trucks are at a depot
        for truck in self.trucks.values():
            if truck.edge is not None or truck.current_node is None:
                return False
            if not self.graph.nodes[truck.current_node].get("has_depot", False):
                return False
                
        return True

    def set_congestion(self, edge: EdgeId, congestion_level: float) -> None:
        """Set congestion for an edge, clamped to the valid ``[0.0, 1.0]`` range."""

        u, v = edge
        if not self.graph.has_edge(u, v):
            raise ValueError(f"edge {edge!r} is not in the graph")
        val = float(np.clip(congestion_level, 0.0, 1.0))
        self.graph.edges[u, v]["congestion_level"] = val
        self._obs_edges[self._edge_key(u, v)]["congestion_level"] = val
        self._get_shortest_path.cache_clear()

    def dispatch_truck(self, truck_id: int, destination: NodeId) -> list[NodeId]:
        """Assign an idle truck to a destination and return the A* route."""

        if truck_id not in self.trucks:
            raise ValueError(f"unknown truck id {truck_id}")
        if destination not in self.graph:
            raise ValueError(f"unknown destination node {destination!r}")

        truck = self.trucks[truck_id]
        if not truck.is_idle:
            raise ValueError(f"truck {truck_id} is already moving")

        path = self._get_shortest_path(truck.current_node, destination)
        truck.destination = destination
        truck.path = path
        truck.path_index = 0
        truck.edge_progress = 0.0
        self._enter_next_edge(truck)
        return path

    @functools.lru_cache(maxsize=None)
    def _get_shortest_path(self, source: NodeId, destination: NodeId) -> list[NodeId]:
        def weight_func(u, v, d):
            return d["distance"] / max(1e-6, 1.0 - d["congestion_level"])
            
        return nx.astar_path(
            self.graph,
            source,
            destination,
            heuristic=self._heuristic,
            weight=weight_func,
        )

    @classmethod
    def from_specs(
        cls,
        nodes: Mapping[NodeId, Mapping[str, Any]],
        edges: Iterable[tuple[NodeId, NodeId, Mapping[str, Any]]],
        **kwargs: Any,
    ) -> GraphEnv:
        """Construct an environment from serializable node and edge specs."""

        return cls(nodes=nodes, edges=edges, **kwargs)

    def _build_graph(
        self,
        graph: nx.Graph | None,
        nodes: Mapping[NodeId, Mapping[str, Any]] | None,
        edges: Iterable[tuple[NodeId, NodeId, Mapping[str, Any]]] | None,
    ) -> nx.Graph:
        if graph is not None and (nodes is not None or edges is not None):
            raise ValueError("pass either graph or nodes/edges specs, not both")

        if graph is None:
            graph = nx.Graph()
            if nodes is None and edges is None:
                nodes = {
                    "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
                    "customer": {"x": 1.0, "y": 0.0, "demand": 1.0, "has_depot": False},
                }
                edges = [("depot", "customer", {"distance": 1.0, "congestion_level": 0.0})]
            for node_id, attrs in (nodes or {}).items():
                graph.add_node(node_id, **dict(attrs))
            for u, v, attrs in (edges or []):
                graph.add_edge(u, v, **dict(attrs))
        else:
            graph = graph.copy()

        if graph.number_of_nodes() == 0:
            raise ValueError("graph must contain at least one node")

        self._normalize_graph_attributes(graph)
        return graph

    def _normalize_graph_attributes(self, graph: nx.Graph) -> None:
        for node, data in graph.nodes(data=True):
            data["x"] = float(data.get("x", 0.0))
            data["y"] = float(data.get("y", 0.0))
            data["demand"] = float(data.get("demand", 0.0))
            data["has_depot"] = bool(data.get("has_depot", False))

        for u, v, data in graph.edges(data=True):
            if "distance" not in data:
                ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
                vx, vy = graph.nodes[v]["x"], graph.nodes[v]["y"]
                data["distance"] = hypot(vx - ux, vy - uy)
            data["distance"] = float(data["distance"])
            if data["distance"] <= 0:
                raise ValueError(f"edge {(u, v)!r} must have positive distance")
            data["congestion_level"] = float(np.clip(data.get("congestion_level", 0.0), 0.0, 1.0))

    def _resolve_depot(self, depot_node: NodeId | None) -> NodeId:
        if depot_node is not None:
            if depot_node not in self.graph:
                raise ValueError(f"depot node {depot_node!r} is not in the graph")
            self.graph.nodes[depot_node]["has_depot"] = True
            return depot_node

        depot_nodes = [node for node, data in self.graph.nodes(data=True) if data["has_depot"]]
        if not depot_nodes:
            raise ValueError("graph must include a depot node or depot_node must be provided")
        return depot_nodes[0]

    def _apply_congestion(
        self,
        congestion_actions: Mapping[EdgeId, float] | Iterable[tuple[NodeId, NodeId, float]] | None,
        info: dict[str, Any],
    ) -> None:
        if congestion_actions is None:
            return

        if isinstance(congestion_actions, Mapping):
            updates = [(u, v, level) for (u, v), level in congestion_actions.items()]
        else:
            updates = list(congestion_actions)

        for u, v, level in updates:
            self.set_congestion((u, v), level)
            info["congestion_updates"].append(
                {"edge": self._edge_key(u, v), "congestion_level": self.graph.edges[u, v]["congestion_level"]}
            )

    def _apply_dispatch(self, dispatch_actions: Mapping[int, NodeId] | None, info: dict[str, Any]) -> None:
        if not dispatch_actions:
            return

        for truck_id, destination in dispatch_actions.items():
            truck = self.trucks.get(truck_id)
            if truck is None or not truck.is_idle:
                info["ignored_dispatches"].append({"truck_id": truck_id, "destination": destination})
                continue
            path = self.dispatch_truck(truck_id, destination)
            if len(path) == 1:
                self._handle_node_stop(truck, destination, info)
                truck.destination = None
            info["dispatched"].append({"truck_id": truck_id, "destination": destination, "path": path})

    def _move_truck_one_tick(self, truck: TruckState, info: dict[str, Any]) -> float:
        remaining_time = 1.0
        travelled = 0.0

        while remaining_time > 1e-12 and truck.edge is not None:
            u, v = truck.edge
            edge_data = self.graph.edges[u, v]
            speed_multiplier = 1.0 - edge_data["congestion_level"]
            effective_speed = truck.speed * speed_multiplier
            if effective_speed <= 1e-12:
                break

            edge_distance = edge_data["distance"]
            remaining_distance = edge_distance - truck.edge_progress
            max_distance_this_tick = effective_speed * remaining_time

            if max_distance_this_tick + 1e-12 < remaining_distance:
                truck.edge_progress += max_distance_this_tick
                travelled += max_distance_this_tick
                remaining_time = 0.0
            else:
                truck.edge_progress = edge_distance
                travelled += remaining_distance
                remaining_time -= remaining_distance / effective_speed
                self._arrive_at_edge_end(truck, info)

        return travelled

    def _enter_next_edge(self, truck: TruckState) -> None:
        if truck.path_index >= len(truck.path) - 1:
            truck.edge = None
            truck.current_node = truck.path[truck.path_index]
            return

        u = truck.path[truck.path_index]
        v = truck.path[truck.path_index + 1]
        truck.current_node = None
        truck.edge = (u, v)
        truck.edge_progress = 0.0

    def _arrive_at_edge_end(self, truck: TruckState, info: dict[str, Any]) -> None:
        truck.path_index += 1
        arrived_node = truck.path[truck.path_index]
        truck.current_node = arrived_node
        truck.edge = None
        truck.edge_progress = 0.0
        info["arrivals"].append({"truck_id": truck.truck_id, "node": arrived_node})

        if arrived_node == truck.destination:
            self._handle_node_stop(truck, arrived_node, info)
            truck.destination = None

        self._enter_next_edge(truck)

    def _handle_node_stop(self, truck: TruckState, node: NodeId, info: dict[str, Any]) -> None:
        if self.graph.nodes[node].get("has_depot", False):
            self._reload_truck(truck, info, node)
        else:
            self._serve_demand(truck, node, info)

    def _reload_truck(self, truck: TruckState, info: dict[str, Any], node: NodeId) -> None:
        if truck.load >= truck.capacity:
            return
        reloaded = truck.capacity - truck.load
        truck.load = truck.capacity
        info["reloads"].append({"truck_id": truck.truck_id, "node": node, "reloaded": reloaded})

    def _serve_demand(self, truck: TruckState, node: NodeId, info: dict[str, Any]) -> None:
        demand = self.graph.nodes[node]["demand"]
        if demand <= 0 or truck.load <= 0:
            return

        delivered = min(truck.load, demand)
        self.graph.nodes[node]["demand"] = demand - delivered
        self._obs_nodes[node]["demand"] = demand - delivered
        self.remaining_demand -= delivered
        truck.load -= delivered
        truck.delivered_total += delivered
        info["deliveries"].append({"truck_id": truck.truck_id, "node": node, "delivered": delivered})

    def _reward(self, info: Mapping[str, Any]) -> float:
        delivered = sum(delivery["delivered"] for delivery in info["deliveries"])
        return float(delivered - self.remaining_demand)

    def _heuristic(self, u: NodeId, v: NodeId) -> float:
        ux, uy = self._initial_graph.nodes[u]["x"], self._initial_graph.nodes[u]["y"]
        vx, vy = self._initial_graph.nodes[v]["x"], self._initial_graph.nodes[v]["y"]
        import math
        # Convert EPSG:4326 degrees to rough meters
        dx = (vx - ux) * 111000.0 * math.cos(math.radians((uy + vy) / 2.0))
        dy = (vy - uy) * 111000.0
        return hypot(dx, dy)

    def _edge_key(self, u: NodeId, v: NodeId) -> EdgeId:
        if self.graph.is_directed():
            return (u, v)
        return (u, v) if repr(u) <= repr(v) else (v, u)
