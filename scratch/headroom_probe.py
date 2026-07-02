"""Measure RL headroom for capacity>1 Stage-0 BEFORE training (fast, cached).

The capacity-1 rung failed because greedy (=SPT) was optimal. With capacity>1 each trip is
a multi-stop tour where greedy nearest-neighbour is a heuristic, not optimal -- so there
*should* be headroom. This probe quantifies it without any learning:

  greedy total_wait   vs.   best fixed-priority policy found by hill-climbing.

A clear gap (greedy >> best_pi) means real headroom; ~0 gap means the geometry still makes
the decision inconsequential. Disk/graph load is cached so thousands of evals run in seconds.
"""

from __future__ import annotations

import copy
import functools
import random

import networkx as nx

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.stage0_factory import _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS, _id_key
from src.utils.graph_utils import load_osm_graph_and_demands
from src.baselines.greedy_dispatch import (
    greedy_protagonist_policy,
    no_antagonist_policy,
    run_episode,
)


@functools.lru_cache(maxsize=1)
def _load():
    nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
    g = nx.Graph()
    for nid, a in nodes.items():
        g.add_node(nid, **a)
    for u, v, a in edges:
        g.add_edge(u, v, **a)
    return nodes, edges, g


def _cluster(hotspot: str, depot: str, cluster_size: int) -> list:
    _, _, g = _load()
    dl = nx.single_source_dijkstra_path_length(g, hotspot, weight="distance")
    ordered = sorted(dl.items(), key=lambda kv: (kv[1], _id_key(kv[0])))
    out = []
    for nid, _ in ordered:
        if nid == depot:
            continue
        out.append(nid)
        if len(out) == cluster_size:
            break
    return out


def fast_factory(depot: str, cluster: list, capacity: float):
    nodes_t, edges, _ = _load()

    def make() -> GraphEnv:
        nodes = copy.deepcopy(nodes_t)
        for n in nodes:
            nodes[n]["demand"] = 0.0
        for n in cluster:
            nodes[n]["demand"] = 1.0
        nodes[depot]["has_depot"] = True
        env = GraphEnv(nodes=nodes, edges=edges, num_trucks=1, truck_capacity=capacity,
                       truck_starting_nodes=[depot], truck_speed=1.0, max_time=800)
        env.stage0_cluster = tuple(cluster)
        return env

    return make


def _cfg() -> SMDPConfig:
    return SMDPConfig(max_ticks=800, reward_mode="latency", antagonist_interval=20,
                      congestion_duration=30, congestion_budget=300.0, congestion_cooldown=0,
                      congestion_cost=0.1, congestion_levels=(0.25, 0.5, 0.75, 1.0))


def priority_policy(priority: dict):
    def policy(event):
        actions = {}
        for tid, dests in event.protagonist_action_mask.items():
            if not dests:
                continue
            customers = [d for d in dests if d in priority]
            actions[tid] = max(customers, key=lambda d: priority[d]) if customers else dests[0]
        return actions
    return policy


def probe(depot: str, cluster_size: int, capacity: float, hotspot: str = "284",
          restarts: int = 4, iters: int = 120):
    cfg = _cfg()
    cluster = _cluster(hotspot, depot, cluster_size)
    make = fast_factory(depot, cluster, capacity)

    smdp = SMDPDecisionWrapper(env_factory=make, config=cfg)
    greedy = run_episode(smdp, greedy_protagonist_policy(smdp), no_antagonist_policy)

    def ev(order):
        pr = {n: len(order) - i for i, n in enumerate(order)}
        s = SMDPDecisionWrapper(env_factory=make, config=cfg)
        return run_episode(s, priority_policy(pr), no_antagonist_policy)["total_wait"]

    rng = random.Random(0)
    best = float("inf")
    for _ in range(restarts):
        order = cluster[:]
        rng.shuffle(order)
        cur = ev(order)
        for _ in range(iters):
            i, j = rng.randrange(len(order)), rng.randrange(len(order))
            order[i], order[j] = order[j], order[i]
            cand = ev(order)
            if cand <= cur:
                cur = cand
            else:
                order[i], order[j] = order[j], order[i]
        best = min(best, cur)

    gap = greedy["total_wait"] - best
    pct = 100 * gap / greedy["total_wait"]
    print(f"depot={depot:>4} cluster={cluster_size:>2} cap={capacity:>3.0f} | "
          f"greedy={greedy['total_wait']:7.0f} best_pi={best:7.0f} | headroom={gap:6.0f} ({pct:4.1f}%) | "
          f"deliv={greedy['delivered']}/{greedy['num_requests']} ticks={greedy['ticks']}")


def spread_demand(depot: str, n: int) -> list:
    """Farthest-point sampling: n geographically spread demand nodes (real TSP structure)."""
    _, _, g = _load()
    apsp = dict(nx.all_pairs_dijkstra_path_length(g, weight="distance"))
    chosen = [depot]
    candidates = [x for x in g.nodes() if x != depot]
    while len(chosen) < n + 1:
        nxt = max(candidates, key=lambda c: (min(apsp[c][s] for s in chosen), _id_key(c)))
        chosen.append(nxt)
        candidates.remove(nxt)
    return chosen[1:]


def probe_spread(depot: str, n: int, capacity: float, restarts: int = 4, iters: int = 150):
    cfg = _cfg()
    cluster = spread_demand(depot, n)
    make = fast_factory(depot, cluster, capacity)
    smdp = SMDPDecisionWrapper(env_factory=make, config=cfg)
    greedy = run_episode(smdp, greedy_protagonist_policy(smdp), no_antagonist_policy)

    def ev(order):
        pr = {nn: len(order) - i for i, nn in enumerate(order)}
        s = SMDPDecisionWrapper(env_factory=make, config=cfg)
        return run_episode(s, priority_policy(pr), no_antagonist_policy)["total_wait"]

    rng = random.Random(0)
    best = float("inf")
    for _ in range(restarts):
        order = cluster[:]
        rng.shuffle(order)
        cur = ev(order)
        for _ in range(iters):
            i, j = rng.randrange(len(order)), rng.randrange(len(order))
            order[i], order[j] = order[j], order[i]
            cand = ev(order)
            if cand <= cur:
                cur = cand
            else:
                order[i], order[j] = order[j], order[i]
        best = min(best, cur)
    gap = greedy["total_wait"] - best
    pct = 100 * gap / greedy["total_wait"]
    print(f"SPREAD depot={depot:>4} n={n:>2} cap={capacity:>3.0f} | "
          f"greedy={greedy['total_wait']:7.0f} best_pi={best:7.0f} | headroom={gap:6.0f} ({pct:4.1f}%) | "
          f"deliv={greedy['delivered']}/{greedy['num_requests']} ticks={greedy['ticks']}")


if __name__ == "__main__":
    import time
    print("=== capacity>1 headroom probe (no antagonist) ===")
    print("--- clustered demand ---")
    for depot, cs, cap in [
        ("39", 8, 1.0),    # the failed rung (sanity: expect ~0%)
        ("39", 8, 4.0),    # far depot, tight cluster
        ("272", 12, 4.0),  # near depot, larger cluster
    ]:
        probe(depot, cs, cap)
    print("--- spread demand, high capacity (minimum-latency tour) ---")
    for depot, n, cap in [
        ("39", 12, 12.0),  # single full tour = pure MLP
        ("39", 12, 6.0),   # two trips
        ("39", 16, 16.0),  # single full tour
        ("39", 16, 8.0),
    ]:
        t = time.time()
        probe_spread(depot, n, cap, restarts=8, iters=400)
        print(f"    ({time.time()-t:.1f}s)")
