#!/usr/bin/env python3
"""Oracle-only probe (no training): is the 'observable threat map' an independent information
source, or a monotone transform of geometry?

Motivation (CRITIQUE_12-07-26.md): every threat map in the programme is produced by
`length_band_vulnerability` (an affine map of edge length into the band), so within one instance
edge vulnerability is by construction perfectly rank-correlated with edge length. This probe
quantifies the consequence at the ROUTE level (the features the generalist head actually reads:
per-route [cost, worst-vulnerability]) across sampled instances of the gen15 pool recipe, and
computes how much the equilibrium moves when the SAME vulnerability values are randomly PERMUTED
across candidate edges (i.e. a threat map decorrelated from geometry). If route cost and route
worst-vuln are strongly correlated and permuted maps shift the equilibrium a lot, then (a) the
map-conditioning claim is partially confounded with geometry-conditioning, and (b) the decisive
eval-only experiment is scoring the frozen generalist against shuffled-map equilibria.

Exact oracle arithmetic; seconds; zero training.
"""
from __future__ import annotations

import random

import numpy as np

from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import solve_multiconvoy


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    insts = sample_instances(8, N=3, K=1, band=(0.15, 0.95), k_extra=8, seed=0)
    print("instance | corr(route cost, route worst-vuln) | eq | eq after edge-permuted map "
          "(mean +/- std over 5 shuffles) | leader-mass L1 shift (mean)")
    for inst in insts:
        g = inst.env.game
        vuln = inst.env.game.edge_vulnerability if hasattr(inst.env.game, "edge_vulnerability") else None
        # per-route features exactly as the head sees them
        costs = [sum(inst.env.graph[u][v]["w"] for u, v in zip(r, r[1:])) for r in g.routes]
        # route worst-vulnerability from the env's observable map
        emap = inst.env.edge_vulnerability
        def route_wv(route):
            vals = []
            for u, v in zip(route, route[1:]):
                vals.append(emap.get((u, v), emap.get((v, u), emap.get(tuple(sorted((u, v), key=repr)), 0.0))))
            return max(vals)
        wv = [route_wv(r) for r in g.routes]
        c = pearson(costs, wv)
        sol = solve_multiconvoy(g, 3, "mission")
        base_eq, base_d = float(sol.loss_mixed), np.asarray(sol.defender_strategy, float)

        # permuted maps: same multiset of vulnerabilities, shuffled across candidate edges
        eqs, l1s = [], []
        cand = sorted({e for r in g.route_edges for e in r}, key=repr)

        def key(e):
            u, v = tuple(e)
            return emap.get((u, v), emap.get((v, u), emap.get(tuple(sorted((u, v), key=repr)))))
        vals = [key(e) for e in cand]
        from src.baselines.interdiction_oracle import build_interdiction_game, survival_intercept_fn
        s_node, t_node = inst.od
        for s in range(5):
            rng = random.Random(1000 + s)
            perm = vals[:]
            rng.shuffle(perm)
            newmap = dict(zip(cand, perm))
            g2 = build_interdiction_game(inst.env.graph, s_node, t_node, 1, k_extra=8,
                                         intercept_fn=survival_intercept_fn(newmap))
            sol2 = solve_multiconvoy(g2, 3, "mission")
            eqs.append(float(sol2.loss_mixed))
            d2 = np.asarray(sol2.defender_strategy, float)
            if d2.shape == base_d.shape:
                l1s.append(float(np.abs(base_d - d2).sum()))
        print(f"{inst.od} R={g.n_routes} | corr={c:+.3f} | eq={base_eq:.3f} | "
              f"{np.mean(eqs):.3f} +/- {np.std(eqs):.3f} | L1={np.mean(l1s):.2f}")


if __name__ == "__main__":
    main()
