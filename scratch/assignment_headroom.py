"""Headroom gate for the 3b assignment probe (before any training).

Question: does greedy-insertion leave room for a better assignment to beat it -- statically
and under attack? If yes, training is justified (RL can learn the better assignment). If
greedy is already near-optimal even under attack, stop and redesign the geometry.

Compares total_wait (sum of unit latencies, lower=better) for:
  greedy-insertion  vs  best fixed-priority assignment (hill-climb)   x   {no attack, attack}.
Policies claim requests sequentially within a decision so two trucks never grab the same node.
"""

from __future__ import annotations

import copy
import functools
import random

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import (
    _DEFAULT_DEMAND, _DEFAULT_DEPOTS, _DEFAULT_EDGES, _DEFAULT_NODES, _DEFAULT_TASKS,
)
from src.utils.graph_utils import load_osm_graph_and_demands
from src.baselines.greedy_dispatch import _congestion_aware_distance, run_episode, no_antagonist_policy


@functools.lru_cache(maxsize=1)
def _load():
    return load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)


def fast_factory(depots, demand_nodes, capacity=1.0, max_time=800):
    nodes_t, edges = _load()

    def make() -> GraphEnv:
        nodes = copy.deepcopy(nodes_t)
        for n in nodes:
            nodes[n]["demand"] = 0.0
        for n in demand_nodes:
            nodes[n]["demand"] = 1.0
        for d in depots:
            nodes[d]["has_depot"] = True
        env = GraphEnv(nodes=nodes, edges=edges, num_trucks=len(depots), truck_capacity=capacity,
                       truck_starting_nodes=list(depots), truck_speed=1.0, max_time=max_time)
        return env
    return make


def _cfg():
    return SMDPConfig(max_ticks=800, reward_mode="latency", routing_mode="destination",
                      antagonist_interval=20, congestion_duration=30, congestion_budget=400.0,
                      congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(0.25, 0.5, 0.75, 1.0))


def greedy_seq_policy(smdp):
    """Greedy-insertion with sequential claiming: each idle truck takes the nearest
    (congestion-aware) unclaimed request; depot only if none."""
    def pol(event):
        env = smdp.env
        actions, claimed = {}, set()
        for tid in sorted(event.protagonist_action_mask):
            dests = [d for d in event.protagonist_action_mask[tid] if d not in claimed]
            if not dests:
                continue
            reqs = [d for d in dests if env.graph.nodes[d]["demand"] > 0]
            src = env.trucks[tid].current_node
            if reqs:
                best = min(reqs, key=lambda d: _congestion_aware_distance(env, src, d))
                actions[tid] = best
                claimed.add(best)
            else:
                actions[tid] = dests[0]
        return actions
    return pol


def priority_seq_policy(smdp, order):
    rank = {n: i for i, n in enumerate(order)}
    def pol(event):
        env = smdp.env
        actions, claimed = {}, set()
        for tid in sorted(event.protagonist_action_mask):
            dests = [d for d in event.protagonist_action_mask[tid] if d not in claimed]
            if not dests:
                continue
            reqs = [d for d in dests if d in rank]
            if reqs:
                best = min(reqs, key=lambda d: rank[d])
                actions[tid] = best
                claimed.add(best)
            else:
                actions[tid] = dests[0]
        return actions
    return pol


def congest_near_trucks_antagonist(smdp):
    """Adaptive adversary: each decision, congest an allowed edge (near a truck) at max level."""
    def pol(event):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        edge = sorted(lbe.keys(), key=repr)[0]
        return (edge, max(lbe[edge]))
    return pol


def best_priority(make, cfg, demand_nodes, antag, restarts=6, iters=200):
    rng = random.Random(0)
    best = float("inf")
    for _ in range(restarts):
        order = list(demand_nodes)
        rng.shuffle(order)
        s = SMDPDecisionWrapper(env_factory=make, config=cfg)
        cur = run_episode(s, priority_seq_policy(s, order), antag(s) if antag else no_antagonist_policy)["total_wait"]
        for _ in range(iters):
            i, j = rng.randrange(len(order)), rng.randrange(len(order))
            order[i], order[j] = order[j], order[i]
            s = SMDPDecisionWrapper(env_factory=make, config=cfg)
            cand = run_episode(s, priority_seq_policy(s, order), antag(s) if antag else no_antagonist_policy)["total_wait"]
            if cand <= cur:
                cur = cand
            else:
                order[i], order[j] = order[j], order[i]
        best = min(best, cur)
    return best


def probe(depots, demand_nodes, label):
    cfg = _cfg()
    make = fast_factory(depots, demand_nodes)
    for atk_label, antag in [("no-attack", None), ("attack", congest_near_trucks_antagonist)]:
        s = SMDPDecisionWrapper(env_factory=make, config=cfg)
        g = run_episode(s, greedy_seq_policy(s), antag(s) if antag else no_antagonist_policy)["total_wait"]
        b = best_priority(make, cfg, demand_nodes, antag)
        gap = g - b
        print(f"  {label:18s} {atk_label:9s} | greedy={g:7.0f} best={b:7.0f} | headroom={gap:6.0f} ({100*gap/g:4.1f}%)")


if __name__ == "__main__":
    print("=== assignment headroom gate (greedy-insertion vs best fixed-priority) ===")
    probe(_DEFAULT_DEPOTS, _DEFAULT_DEMAND, "contested-8")
    # a couple of variants to check geometry sensitivity
    probe(_DEFAULT_DEPOTS, ("237", "78", "130", "27", "49", "224", "43", "220", "46", "47", "48", "225"), "contested-12")
    probe(("110", "135"), ("237", "130", "49", "43", "220", "47"), "contested-6")
