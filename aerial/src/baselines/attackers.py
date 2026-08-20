"""Scripted attackers for the robustness-evaluation portfolio.

These are policy-agnostic disruptors that evaluate every protagonist under the same attacks, unlike
the co-evolved antagonist, which is a best response to whichever policy it trained against. All of
them respect the antagonist action mask (reach and budget), so they are directly comparable to
learned attackers.
"""

from __future__ import annotations

import random
from typing import Any, Callable

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


def mask_first_block_policy(event: DecisionEvent):
    """Deterministic gateway attacker: block the lexicographically-first maskable edge at max level.

    Under route reach the mask is exactly the edges on the trucks' committed routes, so the mask
    itself does the aiming. Held out as a test attacker, never trained against.
    """
    lbe = event.antagonist_action_mask.get("levels_by_edge", {})
    if not lbe:
        return None
    edge = sorted(lbe, key=repr)[0]
    return (edge, max(lbe[edge]))


def random_path_block_policy(smdp: SMDPDecisionWrapper, seed: int = 0):
    """Block the first blockable edge on the path of a uniformly random goal-committed truck.

    Falls back to a random maskable edge when no truck is committed. Route-aimed like `targeted`
    but stochastic across trucks, so training against it leaves less determinism to overfit and
    keeps `targeted` held out as the test attack.
    """
    rng = random.Random(seed)

    def policy(event: DecisionEvent):
        env = smdp.env
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        candidates = []
        for truck in sorted(env.trucks.values(), key=lambda t: t.truck_id):
            goal = getattr(truck, "assigned_target", None) or truck.destination
            start = truck.current_node
            if start is None and truck.edge is not None:
                start = truck.edge[1]
            if goal is None or start is None or start == goal:
                continue
            try:
                path = nx.dijkstra_path(env.graph, start, goal, weight="effective_weight")
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue
            for i in range(len(path) - 1):
                e = env._edge_key(path[i], path[i + 1])
                if e in lbe:
                    candidates.append(e)
                    break  # first blockable edge on THIS truck's path
        edge = rng.choice(candidates) if candidates else rng.choice(sorted(lbe, key=repr))
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


class ScriptedAttackerMixture:
    """A fixed weighted mixture of scripted attackers, one sampled per episode (seeded).

    Training the defender against a mixture rather than a single fixed attacker exposes it to a
    diversity of interdiction patterns and denies it one attacker to overfit. There is no inner
    best-response loop, so the adversary stays cheap and stationary. ``members`` is a list of
    ``(name, policy_callable, weight)`` whose policies are already bound to their wrapper, and
    ``sample()`` returns ``(name, policy)`` for the next episode.
    """

    def __init__(self, members: list[tuple[str, Callable[[DecisionEvent], Any], float]], seed: int = 0):
        if not members:
            raise ValueError("mixture needs at least one member")
        if any(w < 0 for _, _, w in members) or sum(w for _, _, w in members) <= 0:
            raise ValueError("weights must be non-negative and sum to > 0")
        self.names = [n for n, _, _ in members]
        self._policies = [p for _, p, _ in members]
        self._weights = [w for _, _, w in members]
        self._rng = random.Random(seed)
        self.counts = {n: 0 for n in self.names}

    def sample(self) -> tuple[str, Callable[[DecisionEvent], Any]]:
        idx = self._rng.choices(range(len(self._policies)), weights=self._weights, k=1)[0]
        self.counts[self.names[idx]] += 1
        return self.names[idx], self._policies[idx]


def build_scripted_attacker(name: str, smdp: SMDPDecisionWrapper, seed: int = 0):
    """Map an attacker name to a ready policy bound to ``smdp``."""
    if name == "targeted":
        return targeted_block_policy(smdp)
    if name == "pathrand":
        return random_path_block_policy(smdp, seed=seed)
    if name == "gateway":
        return mask_first_block_policy
    if name == "random":
        return random_block_policy(seed=seed)
    raise ValueError(f"unknown scripted attacker {name!r}")
