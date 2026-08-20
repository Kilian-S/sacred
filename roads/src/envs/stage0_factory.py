"""Stage-0 validation-rung environment factory (problem redesign §7.1).

Stage 0 is a *validation rung*, not a research result: its only job is to prove
the (now-stable) SAC+ATLA+GNN stack can learn a **consequential, adversarial**
routing policy at all. It deliberately makes the per-step decision matter, unlike
the diffuse static-demand OSM problem where ``Q_Spread`` collapsed to ~0.

Design (confirmed with Kilian):
  * Same Kaliningrad OSM graph as :func:`src.envs.osm_factory.make_osm_env`
    (so the GNN node/edge feature schema is identical; ``node_in_dim=11`` carries).
  * **1 truck / 1 depot.** **K=1**: a single tight demand cluster placed around the
    densest hotspot node, with the depot ~20 distance units away so there is a real
    corridor for the antagonist to congest.
  * **Capacity-1 shuttle**: the truck carries one request, returns to the depot to
    reload, and repeats. The learnable, consequential decision is therefore the
    **serve order** of the outstanding requests (shortest-processing-time
    scheduling), which the antagonist can attack by congesting the corridor.
  * A short *fixed* set of unit-demand requests (no Poisson arrivals yet — that is
    Stage 1). All requests are present at ``t=0``, so per-request ``arrival_tick`` is
    0 and the latency reward (see ``smdp_wrapper``) telescopes to total wait time.

This factory is **additive**: it does not touch :func:`make_osm_env` or the static
demand path, which remain the documented baseline + ZST reference.
"""

from __future__ import annotations

from pathlib import Path

import networkx as nx

from src.env.graph_env import GraphEnv
from src.utils.graph_utils import load_osm_graph_and_demands

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_DEFAULT_NODES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson")
_DEFAULT_EDGES = str(PROJECT_ROOT / "data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson")
_DEFAULT_TASKS = str(PROJECT_ROOT / "data/maps/koenigsberg1.json")


def make_stage0_env(
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
    *,
    depot: str = "39",
    hotspot: str | None = None,
    cluster_size: int = 8,
    request_size: float = 1.0,
    truck_capacity: float = 1.0,
    truck_speed: float = 1.0,
    max_time: int = 400,
) -> GraphEnv:
    """Build the Stage-0 single-truck validation environment on the OSM graph.

    Parameters
    ----------
    depot:
        Node id used as the single depot / truck home. Defaults to ``"39"`` — a node
        ~20 distance units from the demand hotspot, giving a meaningful corridor.
    hotspot:
        Node id at the centre of the K=1 demand cluster. ``None`` (default) picks the
        node with the highest ``base_demand`` deterministically (tie-break by id).
    cluster_size:
        Number of unit-demand requests = the hotspot plus its ``cluster_size - 1``
        nearest nodes by graph distance (a single tight K=1 region).
    request_size:
        Demand placed on each cluster node (1.0 = a unit request).
    truck_capacity:
        Truck capacity. ``1.0`` (default) forces a depot-reload shuttle, so the
        consequential decision is the serve order of the cluster.
    max_time:
        Episode horizon in ticks. The SMDP wrapper overrides this with
        ``SMDPConfig.max_ticks`` at reset, so this is only a standalone default.
    """
    if cluster_size < 1:
        raise ValueError("cluster_size must be at least 1")

    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)

    # Build a lightweight NetworkX graph purely to resolve the hotspot + cluster by
    # shortest-path distance (deterministic, tie-broken by node id).
    g = nx.Graph()
    for node_id, attrs in nodes.items():
        g.add_node(node_id, **attrs)
    for u, v, attrs in edges:
        g.add_edge(u, v, **attrs)

    if hotspot is None:
        # Densest node by the demand heatmap; tie-break by id for reproducibility.
        hotspot = max(nodes, key=lambda n: (nodes[n].get("base_demand", 0.0), _id_key(n)))
    elif hotspot not in nodes:
        raise ValueError(f"hotspot node {hotspot!r} is not in the graph")
    if depot not in nodes:
        raise ValueError(f"depot node {depot!r} is not in the graph")

    dist_from_hotspot = nx.single_source_dijkstra_path_length(g, hotspot, weight="distance")
    # cluster = hotspot + nearest nodes, excluding the depot so it stays demand-free.
    ordered = sorted(dist_from_hotspot.items(), key=lambda kv: (kv[1], _id_key(kv[0])))
    cluster: list[str] = []
    for node_id, _ in ordered:
        if node_id == depot:
            continue
        cluster.append(node_id)
        if len(cluster) == cluster_size:
            break

    # Place a single fixed K=1 demand cluster; zero everything else.
    for node_id in nodes:
        nodes[node_id]["demand"] = 0.0
    for node_id in cluster:
        nodes[node_id]["demand"] = float(request_size)

    nodes[depot]["has_depot"] = True

    env = GraphEnv(
        nodes=nodes,
        edges=edges,
        num_trucks=1,
        truck_capacity=truck_capacity,
        truck_starting_nodes=[depot],
        truck_speed=truck_speed,
        max_time=max_time,
    )
    # Expose the chosen geometry for logging/debugging (read-only metadata).
    env.stage0_hotspot = hotspot
    env.stage0_depot = depot
    env.stage0_cluster = tuple(cluster)
    return env


def make_stage0_nexthop_env(
    nodes_path: str = _DEFAULT_NODES,
    edges_path: str = _DEFAULT_EDGES,
    tasks_path: str = _DEFAULT_TASKS,
    *,
    depot: str = "14",
    target: str = "82",
    demand: float = 12.0,
    truck_capacity: float = 1.0,
    truck_speed: float = 1.0,
    max_time: int = 400,
) -> GraphEnv:
    """Build the next-hop route-choice validation environment on the OSM graph.

    Focused demand (the redesign's design — *not* spread): all ``demand`` placed on a single
    target node, with a capacity-1 shuttle so the truck makes repeated depot<->target trips,
    each requiring a route choice. ``depot`` and ``target`` are joined by two disjoint routes
    (a shorter "fast" route and a longer "safe" route); the default pair ``14 -> 82`` has two
    edge-disjoint 5-hop routes (fast 6.4 / safe 7.4, ratio 1.16). Greedy defaults to the fast
    route; the robust policy learns to take the safe route when the antagonist congests fast.

    The action model is selected by ``SMDPConfig.routing_mode="next_hop"`` in the wrapper;
    this factory only sets up the geometry. Built on the full OSM graph so the GNN sees the
    real topology. The static OSM problem and :func:`make_stage0_env` are left untouched.
    """
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    if depot not in nodes:
        raise ValueError(f"depot node {depot!r} is not in the graph")
    if target not in nodes:
        raise ValueError(f"target node {target!r} is not in the graph")
    if depot == target:
        raise ValueError("depot and target must differ")

    for node_id in nodes:
        nodes[node_id]["demand"] = 0.0
    nodes[target]["demand"] = float(demand)
    nodes[depot]["has_depot"] = True

    env = GraphEnv(
        nodes=nodes,
        edges=edges,
        num_trucks=1,
        truck_capacity=truck_capacity,
        truck_starting_nodes=[depot],
        truck_speed=truck_speed,
        max_time=max_time,
    )
    env.stage0_depot = depot
    env.stage0_target = target
    return env


def _id_key(node_id: str) -> tuple:
    """Stable sort key for node ids that may be numeric strings ("39" < "113")."""
    s = str(node_id)
    return (0, int(s)) if s.isdigit() else (1, s)
