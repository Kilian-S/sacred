"""Scripted attackers for the robustness-evaluation portfolio.

These are policy-agnostic disruptors used to evaluate every protagonist under the SAME attacks
(unlike the co-evolved antagonist, which is a best response to whichever policy it trained
against). Both respect the antagonist action mask (reach + budget), so they are directly
comparable to learned attackers.

  * random_block_policy  — uniform-random full block from the current mask (seeded). The weakest
    portfolio member: robustness to undirected disruption / domain-randomization-style noise.
  * targeted_block_policy — deterministic heuristic: block the first blockable edge AHEAD on the
    congestion-aware shortest path of the truck nearest to its goal (i.e. cut off the truck about
    to make progress). The strongest scripted member; also the VALIDATION attacker for checkpoint
    selection, keeping the best-response test attackers held out.
"""

from __future__ import annotations

import random
from typing import Any

import networkx as nx

from src.env.smdp_wrapper import DecisionEvent, SMDPDecisionWrapper
from src.baselines.greedy_dispatch import _id_key


def random_block_policy(seed: int = 0):
    """Uniform-random maskable edge at max level, every time the mask allows one (seeded)."""
    rng = random.Random(seed)

    def policy(event: DecisionEvent):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        edge = rng.choice(sorted(lbe, key=repr))
        return (edge, max(lbe[edge]))

    return policy


def targeted_block_policy(smdp: SMDPDecisionWrapper):
    """Block the first blockable edge on the shortest path of the truck closest to its goal.

    Deterministic (ties broken by node id), reach-agnostic: with route reach the candidate edges
    are the path itself; with leashed reach only near-truck path edges are maskable. Falls back to
    the lexicographically-first maskable edge when no truck is committed to a goal.
    """

    def policy(event: DecisionEvent):
        env = smdp.env
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None

        best: tuple[float, tuple, Any] | None = None  # (dist to victim's goal, tiebreak, edge)
        for truck in sorted(env.trucks.values(), key=lambda t: t.truck_id):
            goal = getattr(truck, "assigned_target", None) or truck.destination
            start = truck.current_node
            if start is None and truck.edge is not None:
                start = truck.edge[1]
            if goal is None or start is None or start == goal:
                continue
            try:
                path = nx.dijkstra_path(env.graph, start, goal, weight="effective_weight")
                remaining = nx.dijkstra_path_length(env.graph, start, goal, weight="effective_weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for i in range(len(path) - 1):
                e = env._edge_key(path[i], path[i + 1])
                if e in lbe:
                    cand = (remaining, _id_key(path[i]), e)
                    if best is None or cand < best:
                        best = cand
                    break  # only the FIRST blockable edge on this truck's path

        if best is None:
            edge = sorted(lbe, key=repr)[0]
        else:
            edge = best[2]
        return (edge, max(lbe[edge]))

    return policy
