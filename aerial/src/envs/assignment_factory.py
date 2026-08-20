"""Multi-truck assignment-probe environment factories over the OSM graph. Each places ``n``
depots (one truck each) with a fixed set of demand nodes positioned so that which truck serves
a request genuinely matters. Action is destination mode: the policy picks a request and the env
auto-routes via exact Dijkstra, so assignment is exercised without a new network head.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.env.graph_env import GraphEnv
from src.utils.graph_utils import load_osm_graph_and_demands

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_NODES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson")
_DEFAULT_EDGES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson")
_DEFAULT_TASKS = str(PROJECT_ROOT / "data/maps/koenigsberg1.json")

# Two diameter-endpoint depots and a band of demand nodes roughly equidistant from both, so
# the assignment decision is genuinely non-trivial.
_DEFAULT_DEPOTS = ("110", "135")
_DEFAULT_DEMAND = ("237", "78", "130", "27", "49", "224", "43", "220")

# Hybrid geometry (assignment + next-hop routing): demand sits east of the node-0 hub so Depot
# A's routes funnel through the ('0','1') gateway and Depot B's through different chokepoints,
# keeping both the assignment lever and the routing lever live.
_HYBRID_DEPOTS = ("110", "135")
_HYBRID_DEMAND = ("78", "130", "49", "224", "48", "17", "47", "46")


def make_assignment_env(
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
    *,
    depots: tuple[str, ...] = _DEFAULT_DEPOTS,
    demand_nodes: tuple[str, ...] = _DEFAULT_DEMAND,
    demand_per_node: float = 1.0,
    truck_capacity: float = 1.0,
    truck_speed: float = 1.0,
    max_time: int = 800,
    expose_queue_features: bool | None = None,
) -> GraphEnv:
    """Build the n-depot / n-truck assignment-probe environment on the OSM graph.

    One truck starts at each depot. ``demand_nodes`` each carry ``demand_per_node`` units of
    fixed demand. ``truck_capacity=1`` makes each delivery a depot round-trip, so the decision
    is purely which truck serves which request (assignment), with auto-routing handling paths.
    """
    if len(depots) < 1:
        raise ValueError("need at least one depot")

    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    for node_id in (*depots, *demand_nodes):
        if node_id not in nodes:
            raise ValueError(f"node {node_id!r} is not in the graph")

    for node_id in nodes:
        nodes[node_id]["demand"] = 0.0
    for node_id in demand_nodes:
        nodes[node_id]["demand"] = float(demand_per_node)
    for depot_id in depots:
        nodes[depot_id]["has_depot"] = True

    env = GraphEnv(
        nodes=nodes,
        edges=edges,
        num_trucks=len(depots),
        truck_capacity=truck_capacity,
        truck_starting_nodes=list(depots),  # one truck per depot
        truck_speed=truck_speed,
        max_time=max_time,
        expose_queue_features=expose_queue_features,
    )
    env.assignment_depots = tuple(depots)
    env.assignment_demand = tuple(demand_nodes)
    return env


def make_hybrid_assign_env(
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
    *,
    depots: tuple[str, ...] = _HYBRID_DEPOTS,
    demand_nodes: tuple[str, ...] = _HYBRID_DEMAND,
    truck_capacity: float = 1.0,
    max_time: int = 800,
) -> GraphEnv:
    """The hybrid geometry with static demand.

    Same GraphEnv as the assignment probe but with the chokepoint-optimised depots and demand,
    so routes funnel through the node-0 gateways and the routing lever is exercised. The hybrid
    behaviour itself comes from ``SMDPConfig(routing_mode="hybrid")`` in the wrapper, not the env.
    """
    env = make_assignment_env(
        nodes_path, edges_path, tasks_path,
        depots=depots, demand_nodes=demand_nodes, truck_capacity=truck_capacity, max_time=max_time,
        # Static demand, but ship the ETA/goal-distance observation block anyway: the routing
        # policy needs the same congestion-aware distance information greedy's Dijkstra uses.
        expose_queue_features=True)
    return env


def poisson_arrival_fn(hotspot_nodes: tuple[str, ...], rate: float, size: float = 1.0):
    """Build a `(rng, horizon) -> [(tick, node, size), ...]` Poisson arrival sampler.

    A homogeneous Poisson process on the env clock (inter-arrivals ~ Exponential(1/rate)),
    decoupled from what the trucks are doing. Each request's location is drawn uniformly from
    ``hotspot_nodes``. The arrival process is re-sampled fresh each reset; pass a ``demand_seed``
    to reset() to fix the instance for eval.
    """
    nodes = list(hotspot_nodes)

    def arrival_fn(rng: Any, horizon: int) -> list[tuple[int, str, float]]:
        if horizon <= 0 or rate <= 0 or not nodes:
            return []
        arrivals: list[tuple[int, str, float]] = []
        t = 0.0
        while True:
            t += float(rng.exponential(1.0 / rate))
            if t >= horizon:
                break
            node = nodes[int(rng.integers(len(nodes)))]
            arrivals.append((int(t) + 1, node, size))
        return arrivals

    return arrival_fn


def make_dynamic_assign_env(
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
    *,
    depots: tuple[str, ...] = _DEFAULT_DEPOTS,
    hotspot_nodes: tuple[str, ...] = _DEFAULT_DEMAND,
    arrival_rate: float = 0.02,
    truck_capacity: float = 1.0,
    truck_speed: float = 1.0,
    max_time: int = 800,
    demand_seed: int | None = None,
    arrival_schedule: list[tuple[int, str, float]] | None = None,
) -> GraphEnv:
    """The contested multi-truck assignment geometry made dynamic.

    Same depots and hotspot band as :func:`make_assignment_env`, but demand is zero at t=0 and
    arrives over the horizon on a Poisson process (rate ``arrival_rate`` requests per tick), so
    the queue can build and antagonist congestion compounds the backlog. Destination-mode
    assignment, capacity 1.

    Args:
        arrival_schedule: if given, replay this exact list of ``(tick, node, size)`` arrivals
            instead of sampling, so a twin env can reproduce a real episode's arrivals exactly.
            The arrival function then ignores its rng.
    """
    if len(depots) < 1:
        raise ValueError("need at least one depot")

    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    for node_id in (*depots, *hotspot_nodes):
        if node_id not in nodes:
            raise ValueError(f"node {node_id!r} is not in the graph")

    for node_id in nodes:
        nodes[node_id]["demand"] = 0.0  # demand arrives dynamically, not at t=0
    for depot_id in depots:
        nodes[depot_id]["has_depot"] = True

    if arrival_schedule is not None:
        _replay = list(arrival_schedule)
        arrival_fn = lambda rng, horizon: _replay  # deterministic replay (rng/horizon ignored)
    else:
        arrival_fn = poisson_arrival_fn(hotspot_nodes, arrival_rate)

    env = GraphEnv(
        nodes=nodes,
        edges=edges,
        num_trucks=len(depots),
        truck_capacity=truck_capacity,
        truck_starting_nodes=list(depots),
        truck_speed=truck_speed,
        max_time=max_time,
        demand_arrival_fn=arrival_fn,
        demand_seed=demand_seed,
    )
    env.assignment_depots = tuple(depots)
    env.assignment_demand = tuple(hotspot_nodes)
    env.dynamic_hotspots = tuple(hotspot_nodes)
    env.arrival_rate = float(arrival_rate)
    return env
