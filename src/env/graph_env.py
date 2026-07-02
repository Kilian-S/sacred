"""Headless graph simulation for the SACRED SDVRP environment.

The PettingZoo wrapper and renderer should sit on top of this module. This
file owns only the fast tick-by-tick state transitions: graph attributes,
truck dispatching, routing, congestion updates, and movement physics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import functools
from math import hypot
from typing import Any, Callable, Hashable, Iterable, Mapping

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
    path: tuple[NodeId, ...] = field(default_factory=tuple)
    path_index: int = 0
    edge: EdgeId | None = None
    edge_progress: float = 0.0
    capacity: float = 1.0
    load: float = 1.0
    delivered_total: float = 0.0
    path_edges: set[EdgeId] = field(default_factory=set)
    # Hybrid mode (assignment + next-hop routing): the request the truck was ASSIGNED to serve
    # (then flips to home_depot for the return leg). None in destination/next_hop modes, so the
    # hybrid serve/reload transitions in the env are no-ops there. The policy routes next-hop
    # toward this target; on serving it (load->0) it becomes home_depot; on reloading it clears.
    assigned_target: NodeId | None = None

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
        demand_arrival_fn: Callable[[Any, int], Iterable[tuple[int, NodeId, float]]] | None = None,
        demand_seed: int | None = None,
        expose_queue_features: bool | None = None,
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

        # Dynamic (Poisson) demand: when an arrival fn is supplied the env injects requests over
        # time (Stage 1.5) instead of placing all demand at t=0. The static path is untouched
        # (fn is None -> _dynamic_demand False). The arrival fn is `(rng, horizon) -> iterable of
        # (tick, node, size)`; it is re-drawn each reset so episodes are fresh (or reproducible
        # via demand_seed for the multi-instance eval harness).
        self._demand_arrival_fn = demand_arrival_fn
        self._dynamic_demand = demand_arrival_fn is not None
        # Whether observe() ships the queue/ETA feature block (node_waits, truck_etas,
        # goal_dists). Defaults to dynamic-only (the Stage-1.5 behaviour); the hybrid rung
        # turns it on for STATIC demand too so the policy has the same congestion-aware
        # distance information the greedy baseline's Dijkstra uses (information parity).
        self._expose_queue_features = (
            bool(expose_queue_features) if expose_queue_features is not None else self._dynamic_demand
        )
        self._demand_rng = np.random.default_rng(demand_seed)
        self._arrival_schedule: list[tuple[int, NodeId, float]] = []
        self._arrival_index = 0
        self._pending_arrivals: dict[NodeId, deque] = {}
        self._delivered_latencies: list[int] = []
        # Congestion-aware single-source distance cache for the per-truck ETA feature. Versioned
        # by `_congestion_version` (bumped on every set_congestion) so ETAs always reflect current
        # congestion; cached per source node within a version (a truck idling between congestion
        # changes is free).
        self._congestion_version = 0
        self._ss_cache: dict[NodeId, dict[NodeId, float]] = {}
        self._ss_cache_version = -1

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

    def reset(self, *, demand_seed: int | None = None) -> dict[str, Any]:
        """Reset the simulation clock and return the initial observation.

        ``demand_seed`` (dynamic mode only) makes the Poisson arrival schedule reproducible — used
        by the multi-instance evaluation harness to average over fixed demand instances.
        """

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
                path=(self.truck_starting_nodes[truck_id],),
            )
            for truck_id in range(self.num_trucks)
        }
        
        # Track global demand for O(1) is_done() checks
        self.remaining_demand = sum(data["demand"] for _, data in self.graph.nodes(data=True))
        
        self.valid_customers_by_comp: dict[int, dict[NodeId, float]] = {}
        for n, data in self.graph.nodes(data=True):
            if not data.get("has_depot", False):
                node_demand = float(data.get("demand", 0.0))
                if node_demand > 0.0:
                    comp = self.node_to_component.get(n)
                    if comp not in self.valid_customers_by_comp:
                        self.valid_customers_by_comp[comp] = {}
                    self.valid_customers_by_comp[comp][n] = node_demand
        
        self._idle_trucks_at_depot = sum(
            1 for truck_id in range(self.num_trucks)
            if self.graph.nodes[self.truck_starting_nodes[truck_id]].get("has_depot", False)
        )
        
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
        
        self._node_coords = {
            node: (data["x"], data["y"])
            for node, data in self.graph.nodes(data=True)
        }

        self._arrival_index = 0
        self._pending_arrivals = {}
        self._delivered_latencies = []
        if self._dynamic_demand:
            if demand_seed is not None:
                self._demand_rng = np.random.default_rng(demand_seed)
            horizon = int(self.max_time) if self.max_time is not None else 0
            schedule = list(self._demand_arrival_fn(self._demand_rng, horizon))
            self._arrival_schedule = sorted(schedule, key=lambda r: (r[0], repr(r[1])))
        else:
            self._arrival_schedule = []

        return self.observe()

    def step(
        self,
        dispatch_actions: Mapping[int, NodeId] | None = None,
        congestion_actions: Mapping[EdgeId, float] | Iterable[tuple[NodeId, NodeId, float]] | None = None,
        next_hop_dispatch: Mapping[int, NodeId] | None = None,
    ) -> StepResult:
        """Advance the environment by one tick.

        ``dispatch_actions`` maps idle truck ids to destination node ids (the truck
        follows the A* shortest path there over many ticks).
        ``next_hop_dispatch`` maps idle truck ids to an *adjacent* node and moves the
        truck along that single direct edge (no A*); used by next-hop routing mode so the
        policy — not the pathfinder — chooses the route. The two dispatch modes are
        mutually exclusive in practice.
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
            "demand_arrivals": [],
            "distance_travelled": 0.0,
        }

        self._inject_demand_arrivals(info)
        self._apply_congestion(congestion_actions, info)
        if info["congestion_updates"]:
            self._get_shortest_path.cache_clear()
            
        self._apply_dispatch(dispatch_actions, info)
        self._apply_next_hop_dispatch(next_hop_dispatch, info)

        active_trucks = [t for t in self.trucks.values() if t.edge is not None]
        if active_trucks:
            n = len(active_trucks)
            rem_times = np.ones(n, dtype=np.float64)
            travelled = np.zeros(n, dtype=np.float64)
            
            while True:
                active_mask = rem_times > 1e-12
                for i in range(n):
                    if active_mask[i] and active_trucks[i].edge is None:
                        active_mask[i] = False
                
                if not active_mask.any():
                    break
                    
                eff_speeds = np.zeros(n, dtype=np.float64)
                rem_dists = np.zeros(n, dtype=np.float64)
                
                for i in range(n):
                    if active_mask[i]:
                        u, v = active_trucks[i].edge
                        data = self.graph.edges[u, v]
                        eff_speeds[i] = active_trucks[i].speed * (1.0 - data["congestion_level"])
                        rem_dists[i] = data["distance"] - active_trucks[i].edge_progress
                        
                active_mask &= (eff_speeds > 1e-12)
                if not active_mask.any():
                    break
                    
                max_dists = eff_speeds * rem_times
                not_arrived = active_mask & ((max_dists + 1e-12) < rem_dists)
                arrived = active_mask & ~not_arrived
                
                if not_arrived.any():
                    idx = np.where(not_arrived)[0]
                    for i in idx:
                        active_trucks[i].edge_progress += max_dists[i]
                        travelled[i] += max_dists[i]
                        rem_times[i] = 0.0
                        
                if arrived.any():
                    idx = np.where(arrived)[0]
                    sorted_idx = sorted(idx, key=lambda x: active_trucks[x].truck_id)
                    for i in sorted_idx:
                        t = active_trucks[i]
                        u, v = t.edge
                        edge_dist = self.graph.edges[u, v]["distance"]
                        t.edge_progress = edge_dist
                        travelled[i] += rem_dists[i]
                        rem_times[i] -= rem_dists[i] / eff_speeds[i]
                        self._arrive_at_edge_end(t, info)
                        
            info["distance_travelled"] += float(travelled.sum())

        self.time += 1
        reward = self._reward(info)
        done = self.is_done()
        return StepResult(self.observe(), reward, done, info)

    def observe(self) -> dict[str, Any]:
        """Return a Python-dict observation suitable for wrappers to transform."""
        obs = {
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
                    "path": truck.path,
                    "path_index": truck.path_index,
                    "delivered_total": truck.delivered_total,
                    "assigned_target": truck.assigned_target,
                }
                for truck_id, truck in self.trucks.items()
            },
        }
        if self._expose_queue_features:
            obs["node_waits"], obs["truck_etas"] = self._dynamic_node_features()
            obs["goal_dists"] = self._goal_distances()
        return obs

    def _single_source_lengths(self, source: NodeId) -> dict[NodeId, float]:
        """Congestion-aware single-source shortest-path lengths from ``source``, cached per source
        within a congestion version (invalidated whenever any edge's congestion changes)."""
        if self._ss_cache_version != self._congestion_version:
            self._ss_cache = {}
            self._ss_cache_version = self._congestion_version
        cached = self._ss_cache.get(source)
        if cached is None:
            cached = nx.single_source_dijkstra_path_length(self.graph, source, weight="effective_weight")
            self._ss_cache[source] = cached
        return cached

    def _dynamic_node_features(self) -> tuple[dict[NodeId, float], dict[int, dict[NodeId, float]]]:
        """Per-node oldest wait and per-truck congestion-aware ETAs for the Step-2 observation.

        ``node_waits[node]`` = self.time − oldest pending arrival tick at that node (request age).
        ``truck_etas[truck_id][node]`` = congestion-aware distance from each idle/at-node truck's
        position to every outstanding-demand node and to its home depot (the candidate targets).
        """
        node_waits: dict[NodeId, float] = {}
        for node, dq in self._pending_arrivals.items():
            if dq:
                node_waits[node] = float(self.time - dq[0])

        demand_nodes = [n for customers in self.valid_customers_by_comp.values() for n in customers]
        truck_etas: dict[int, dict[NodeId, float]] = {}
        for truck_id, truck in self.trucks.items():
            src = truck.current_node
            if src is None:
                continue  # mid-edge: no decision pending, ETA not needed
            lengths = self._single_source_lengths(src)
            targets = demand_nodes + ([truck.home_depot] if truck.home_depot is not None else [])
            truck_etas[truck_id] = {n: lengths[n] for n in targets if n in lengths}
        return node_waits, truck_etas

    def _goal_distances(self) -> dict[int, dict[NodeId, float]]:
        """Per-truck congestion-aware distance-to-goal field: for each truck committed to an
        ``assigned_target`` (hybrid mode), the distance from EVERY node to that goal. This is the
        global routing information the 2-layer GNN cannot propagate itself (receptive field 2 hops
        vs graph diameter ~44): with it, each next-hop candidate carries its goal-progress under
        current congestion — parity with the greedy baseline's Dijkstra. Cached per congestion
        version via _single_source_lengths (undirected graph -> from-goal == to-goal)."""
        goal_dists: dict[int, dict[NodeId, float]] = {}
        for truck_id, truck in self.trucks.items():
            goal = truck.assigned_target
            if goal is None or goal not in self.graph:
                continue
            goal_dists[truck_id] = self._single_source_lengths(goal)
        return goal_dists



    def is_done(self) -> bool:
        """Return true when all demand is served or the optional horizon ends."""

        if self.max_time is not None and self.time >= self.max_time:
            return True

        # Dynamic demand keeps arriving until the horizon, so an empty queue is only a lull, not
        # termination — terminate strictly on max_time (handled above), never on remaining==0.
        if self._dynamic_demand:
            return False

        if self.remaining_demand > 0:
            return False
            
        return self._idle_trucks_at_depot == self.num_trucks

    def set_congestion(self, edge: EdgeId, congestion_level: float) -> None:
        """Set congestion for an edge, clamped to the valid ``[0.0, 1.0]`` range."""

        u, v = edge
        if not self.graph.has_edge(u, v):
            raise ValueError(f"edge {edge!r} is not in the graph")
        val = float(np.clip(congestion_level, 0.0, 1.0))
        self.graph.edges[u, v]["congestion_level"] = val
        self.graph.edges[u, v]["effective_weight"] = self.graph.edges[u, v]["distance"] / max(1e-6, 1.0 - val)
        self._obs_edges[self._edge_key(u, v)]["congestion_level"] = val
        self._congestion_version += 1  # invalidate the congestion-aware ETA distance cache

    def dispatch_truck(self, truck_id: int, destination: NodeId) -> list[NodeId]:
        """Assign an idle truck to a destination and return the A* route."""

        if truck_id not in self.trucks:
            raise ValueError(f"unknown truck id {truck_id}")
        if destination not in self.graph:
            raise ValueError(f"unknown destination node {destination!r}")

        truck = self.trucks[truck_id]
        if not truck.is_idle:
            raise ValueError(f"truck {truck_id} is already moving")

        if truck.current_node is not None and self.graph.nodes[truck.current_node].get("has_depot", False):
            self._idle_trucks_at_depot -= 1

        path = self._get_shortest_path(truck.current_node, destination)
        truck.destination = destination
        truck.path = tuple(path)
        truck.path_index = 0
        truck.edge_progress = 0.0
        truck.path_edges = {self._edge_key(path[i], path[i+1]) for i in range(len(path)-1)}
        self._enter_next_edge(truck)
        return path

    def dispatch_truck_edge(self, truck_id: int, neighbor: NodeId) -> None:
        """Move an idle truck one step along the *direct* edge to ``neighbor`` (no A*).

        Unlike :meth:`dispatch_truck`, this commits the truck to the exact (current,
        neighbor) edge and does not reroute around congestion — that is the point of
        next-hop routing: the policy chooses the edge and bears its congestion. The truck
        arrives next tick, becoming idle for the following decision (serving/reloading via
        the usual :meth:`_arrive_at_edge_end` path when ``neighbor`` carries demand/depot).
        """
        if truck_id not in self.trucks:
            raise ValueError(f"unknown truck id {truck_id}")
        truck = self.trucks[truck_id]
        if not truck.is_idle:
            raise ValueError(f"truck {truck_id} is already moving")
        current = truck.current_node
        if current is None:
            raise ValueError(f"truck {truck_id} has no current node")
        if not self.graph.has_edge(current, neighbor):
            raise ValueError(f"node {neighbor!r} is not adjacent to {current!r}")

        if self.graph.nodes[current].get("has_depot", False):
            self._idle_trucks_at_depot -= 1

        truck.destination = neighbor
        truck.path = (current, neighbor)
        truck.path_index = 0
        truck.edge_progress = 0.0
        truck.path_edges = {self._edge_key(current, neighbor)}
        self._enter_next_edge(truck)

    def _apply_next_hop_dispatch(self, next_hop_dispatch: Mapping[int, NodeId] | None, info: dict[str, Any]) -> None:
        if not next_hop_dispatch:
            return
        for truck_id, neighbor in next_hop_dispatch.items():
            truck = self.trucks.get(truck_id)
            if truck is None or not truck.is_idle:
                info["ignored_dispatches"].append({"truck_id": truck_id, "destination": neighbor})
                continue
            self.dispatch_truck_edge(truck_id, neighbor)
            info["dispatched"].append({"truck_id": truck_id, "destination": neighbor, "path": [truck.path[0], neighbor]})

    @functools.lru_cache(maxsize=None)
    def _get_shortest_path(self, source: NodeId, destination: NodeId) -> list[NodeId]:
        # Dijkstra (exact). A* with the lat/lon coordinate heuristic is not reliably admissible
        # on this OSM graph (clamped/rounded edge weights vs degree coords -> paths up to ~140%
        # suboptimal in testing), which would corrupt truck routing and the greedy baseline's
        # ETAs. The graph is small (~290 nodes), so exact Dijkstra is cheap and worth the rigor.
        return nx.dijkstra_path(self.graph, source, destination, weight="effective_weight")

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
            data["effective_weight"] = data["distance"] / max(1e-6, 1.0 - data["congestion_level"])

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

    def _inject_demand_arrivals(self, info: dict[str, Any]) -> None:
        """Inject Poisson demand requests scheduled to arrive by this tick (dynamic mode only).

        Requests are added to the existing per-node ``demand`` machinery (so the action masks,
        valid-customer index, and the potential-based latency reward all work unchanged), and
        their arrival ticks are queued FIFO per node for per-request latency accounting.
        """
        if not self._dynamic_demand:
            return
        now = self.time + 1
        schedule = self._arrival_schedule
        i = self._arrival_index
        n = len(schedule)
        while i < n and schedule[i][0] <= now:
            _tick, node, size = schedule[i]
            i += 1
            if node not in self.graph or self.graph.nodes[node].get("has_depot", False):
                continue
            data = self.graph.nodes[node]
            data["demand"] += size
            self._obs_nodes[node]["demand"] = data["demand"]
            self.remaining_demand += size
            comp = self.node_to_component.get(node)
            self.valid_customers_by_comp.setdefault(comp, {})[node] = data["demand"]
            dq = self._pending_arrivals.setdefault(node, deque())
            for _ in range(int(round(size))):
                dq.append(now)
            info["demand_arrivals"].append({"node": node, "tick": now, "size": size})
        self._arrival_index = i

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
            self._idle_trucks_at_depot += 1
            self._reload_truck(truck, info, node)
        else:
            self._serve_demand(truck, node, info)

    def _reload_truck(self, truck: TruckState, info: dict[str, Any], node: NodeId) -> None:
        # Hybrid: arriving at the depot ends the current assignment even at FULL load (a truck
        # sent home because no unclaimed request remained must become assignable again, else it
        # would orbit the depot forever). No-op in other modes.
        if truck.assigned_target is not None:
            truck.assigned_target = None
        if truck.load >= truck.capacity:
            return
        reloaded = truck.capacity - truck.load
        truck.load = truck.capacity
        info["reloads"].append({"truck_id": truck.truck_id, "node": node, "reloaded": reloaded})

    def _serve_demand(self, truck: TruckState, node: NodeId, info: dict[str, Any]) -> None:
        demand = self.graph.nodes[node]["demand"]
        if demand <= 0 or truck.load <= 0:
            return
        # Hybrid: a truck serves ONLY the request it was assigned (keeps the assignment decision
        # meaningful) — skip demand it merely passes through. No-op in destination/next_hop modes.
        if truck.assigned_target is not None and node != truck.assigned_target:
            return

        delivered = min(truck.load, demand)
        new_demand = demand - delivered
        self.graph.nodes[node]["demand"] = new_demand
        self._obs_nodes[node]["demand"] = new_demand
        self.remaining_demand -= delivered
        truck.load -= delivered
        truck.delivered_total += delivered
        # Hybrid: served the assigned request (load now 0) -> head home to reload.
        if truck.assigned_target is not None and truck.load <= 0:
            truck.assigned_target = truck.home_depot
        delivery_record = {"truck_id": truck.truck_id, "node": node, "delivered": delivered}
        if self._dynamic_demand:
            dq = self._pending_arrivals.get(node)
            now = self.time + 1
            latencies = []
            for _ in range(int(round(delivered))):
                if dq:
                    latencies.append(now - dq.popleft())
            if latencies:
                self._delivered_latencies.extend(latencies)
                delivery_record["latencies"] = latencies
        info["deliveries"].append(delivery_record)

        comp = self.node_to_component.get(node)
        if hasattr(self, 'valid_customers_by_comp') and comp in self.valid_customers_by_comp:
            if new_demand <= 0.0:
                self.valid_customers_by_comp[comp].pop(node, None)
            else:
                self.valid_customers_by_comp[comp][node] = new_demand

        # Hybrid: if the demand here is exhausted, release any OTHER truck still assigned to it
        # (cross-event double assignment) so it can be re-assigned instead of orbiting a
        # zero-demand node forever — assigned_target otherwise only clears on serve/reload.
        if new_demand <= 0.0:
            for other in self.trucks.values():
                if other is not truck and other.assigned_target == node:
                    other.assigned_target = None

    def _reward(self, info: Mapping[str, Any]) -> float:
        delivered = sum(delivery["delivered"] for delivery in info["deliveries"])
        return float(delivered - self.remaining_demand)

    def _heuristic(self, u: NodeId, v: NodeId) -> float:
        ux, uy = self._node_coords[u]
        vx, vy = self._node_coords[v]
        import math
        # Convert EPSG:4326 degrees to rough meters, then to edge-weight units. OSM edge
        # weights are length_m / 100 (see graph_utils), so the heuristic must also be
        # straight-line-metres / 100 to stay <= true path cost. The previous version returned
        # raw metres (~100x the edge scale) -> grossly inadmissible -> A* could return
        # suboptimal paths. /100.0 makes it admissible AND tight.
        dx = (vx - ux) * 111000.0 * math.cos(math.radians((uy + vy) / 2.0))
        dy = (vy - uy) * 111000.0
        return hypot(dx, dy) / 100.0

    def _edge_key(self, u: NodeId, v: NodeId) -> EdgeId:
        if self.graph.is_directed():
            return (u, v)
        return (u, v) if repr(u) <= repr(v) else (v, u)
