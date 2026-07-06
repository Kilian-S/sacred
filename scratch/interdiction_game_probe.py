#!/usr/bin/env python3
"""Feasibility proof for the REDESIGN: convoy routing as a network-interdiction security game.

The inherited congestion problem was structurally unwinnable (observable + reroutable + reversible
=> reactive dominance, flat attack landscape, adversarial RL worth ~0). Application 1's real threat
is INTERDICTION: hidden, irreversible, and PRE-COMMITTED. That is a Stackelberg security game, where
a deterministic router is maximally exploitable and the minimax MIXED strategy provably cuts
interception. This probe computes, on a clean synthetic network and on the real Kaliningrad graph:

  * loss_det   = interception prob of a DETERMINISTIC router (best single path), worst-cased over the
                 attacker's committed interdiction (= what shortest-path / greedy / a collapsed
                 vanilla-SAC policy gets: fully exploitable).
  * loss_mixed = the minimax value: the best MIXED-strategy router vs the best-responding interdictor
                 (= what SACRED, learning the equilibrium via SAC entropy + ATLA, should approach).

A large (loss_det - loss_mixed) gap = the positive thesis result is STRUCTURAL, not hoped-for:
adversarial training buys robustness the deterministic baseline cannot. Swept over interdiction
budget K (number of edges the enemy can cover).

Run: PYTHONPATH=. .venv/bin/python scratch/interdiction_game_probe.py
"""

from __future__ import annotations

import itertools

import networkx as nx
import numpy as np
from scipy.optimize import linprog


def matrix_game_value(payoff: np.ndarray) -> tuple[float, np.ndarray]:
    """Zero-sum matrix game: ROW (defender) MINIMISES expected payoff, COL (attacker) maximises.
    payoff[i,j] = loss when defender plays row i, attacker plays col j. Returns (value, defender
    mixed strategy) via LP: min v s.t. sum_i x_i payoff[i,j] <= v for all j; sum x_i = 1; x>=0."""
    n, m = payoff.shape
    # vars = [x_0..x_{n-1}, v]; minimise v
    c = np.zeros(n + 1); c[-1] = 1.0
    # for each column j: sum_i payoff[i,j] x_i - v <= 0
    A_ub = np.hstack([payoff.T, -np.ones((m, 1))])
    b_ub = np.zeros(m)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    b_eq = np.array([1.0])
    bounds = [(0, 1)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return float(res.x[-1]), res.x[:n]


def interdiction_game(paths: list[list], edges_of: list[set], all_edges: list, K: int):
    """Defender pure strategies = paths (as edge sets). Attacker pure strategies = every K-subset of
    the edges that appear on some path. payoff = 1 if the path shares any edge with the interdiction
    set (intercepted), else 0. Returns (loss_det, loss_mixed)."""
    cand_edges = sorted(set().union(*edges_of), key=repr)
    attacker_sets = list(itertools.combinations(cand_edges, K)) if K <= len(cand_edges) else [tuple(cand_edges)]
    payoff = np.zeros((len(paths), len(attacker_sets)))
    for i, es in enumerate(edges_of):
        for j, aset in enumerate(attacker_sets):
            payoff[i, j] = 1.0 if es & set(aset) else 0.0
    # deterministic defender: it will pick the least-exploitable single path, but the attacker
    # commits to the best response to whatever single path it plays -> min_i max_j payoff[i,j].
    loss_det = float(min(payoff[i].max() for i in range(len(paths))))
    loss_mixed, _ = matrix_game_value(payoff)
    return loss_det, loss_mixed


def k_shortest_paths(G, s, t, k, weight="w"):
    from networkx.algorithms.simple_paths import shortest_simple_paths
    gen = shortest_simple_paths(G, s, t, weight=weight)
    out = []
    for i, p in enumerate(gen):
        if i >= k:
            break
        out.append(p)
    return out


def edgeset(path):
    return {frozenset((path[i], path[i + 1])) for i in range(len(path) - 1)}


def route_set(G, s, t, kpaths, weight="w"):
    """Defender's candidate routes: the edge-DISJOINT paths (the max-min-cut worth of genuinely
    alternative routes: the strategic diversity a convoy planner would use) plus k-shortest paths
    for realism. k-shortest alone are near-duplicates sharing the cheap chokepoint, which
    understates the mixing room; disjoint paths are the ones that matter for interdiction."""
    paths = [list(p) for p in nx.edge_disjoint_paths(G, s, t)]
    seen = {tuple(p) for p in paths}
    for p in k_shortest_paths(G, s, t, kpaths, weight):
        if tuple(p) not in seen:
            paths.append(p)
    return paths


def run(name, G, s, t, kpaths, weight="w"):
    paths = route_set(G, s, t, kpaths, weight)
    edges_of = [edgeset(p) for p in paths]
    all_edges = sorted(set().union(*edges_of), key=repr)
    print(f"\n=== {name}: {len(paths)} candidate routes {s}->{t}, {len(all_edges)} distinct edges ===")
    print(f"{'K (interdictors)':>16} | {'loss_det':>8} | {'loss_mixed':>10} | {'gap':>6} | reading")
    print("-" * 74)
    for K in (1, 2, 3):
        ld, lm = interdiction_game(paths, edges_of, all_edges, K)
        gap = ld - lm
        reading = ("adversarial routing cuts interception "
                   f"{ld*100:.0f}%->{lm*100:.0f}%" if gap > 0.05 else "little room")
        print(f"{K:>16} | {ld:>8.2f} | {lm:>10.2f} | {gap:>6.2f} | {reading}")


def synthetic():
    # base S -> FOB T with 4 routes: 3 edge-disjoint + 1 sharing a chokepoint with R1.
    G = nx.Graph()
    routes = {"R1": ["S", "A", "T"], "R2": ["S", "B", "T"], "R3": ["S", "C", "T"],
              "R4": ["S", "A", "D", "T"]}
    for p in routes.values():
        for i in range(len(p) - 1):
            G.add_edge(p[i], p[i + 1], w=1.0)
    return G


def main():
    run("SYNTHETIC contested corridor", synthetic(), "S", "T", kpaths=8)

    # Real Kaliningrad graph: base = depot 110, FOB = depot 135 (diameter endpoints), top-k routes.
    try:
        from src.envs.assignment_factory import _DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS
        from src.utils.graph_utils import load_osm_graph_and_demands
        nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
        G = nx.Graph()
        for u, v, d in edges:  # edges is a list of (u, v, attrs)
            G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
        run("KALININGRAD 110->135 (disjoint+kshortest, edge-conn 3)", G, "110", "135", kpaths=8)
        run("KALININGRAD 33->71 (high connectivity, edge-conn 6)", G, "33", "71", kpaths=8)
    except Exception as e:
        print(f"\n(Kaliningrad run skipped: {e})")

    print("\nInterpretation: a positive gap = a DETERMINISTIC router is far more exploitable than the")
    print("minimax MIXED router. SACRED (SAC entropy -> mixed strategy, ATLA -> best-response attacker)")
    print("learns toward loss_mixed; shortest-path / vanilla-collapsed policies sit at loss_det. The")
    print("gap is the headline positive result, and loss_mixed is a computable ground-truth to validate against.")


if __name__ == "__main__":
    main()
