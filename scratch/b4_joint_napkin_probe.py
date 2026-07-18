#!/usr/bin/env python3
"""B4 follow-up (ORACLE-ONLY, 2026-07-18, Critic Aerial): does a NAPKIN-LEVEL COORDINATED rule
close the multi-OD correlation gap?

The B4 probe measured v_indep (best product of per-convoy mixtures) vs v_joint (exact joint
minimax): median gap 14.4%. But the naive class is NOT limited to independent rules: a
practitioner can write a COORDINATED napkin rule ("flip a coin between hand-picked deconflicted
joint plans"). Before any two-target/multi-OD register claims a class-separation moat, the
ladder must carry:

  * deconflict-uniform (payoff-BLIND): uniform over all route PAIRS with zero shared edges
    (fallback: minimum-overlap pairs) - needs only the route lists;
  * best m-pairing uniform mixture (IN-SAMPLE cap, m=2 exhaustive, m<=4 greedy-from-top):
    the upper bound on every "mix over a few hand-built joint plans" rule, disclosed as
    oracle-fit (the robust-static analogue).

Same triples as scratch/b4_multiod_probe.py (seed 11, same filters), same payoff tensor.
"""
from __future__ import annotations

import itertools
import json
import random

import networkx as nx
import numpy as np

from scratch.b4_multiod_probe import build_graph, payoff_tensor, best_product
from src.baselines.interdiction_oracle import build_route_set, edges_of_route
from src.baselines.multiconvoy_oracle import _row_minimiser

KX = 8


def deconflict_uniform(M, routes1, routes2):
    """Payoff-blind napkin rule: uniform over the zero-overlap (else min-overlap) route pairs."""
    e1 = [edges_of_route(r) for r in routes1]
    e2 = [edges_of_route(r) for r in routes2]
    ov = np.array([[len(a & b) for b in e2] for a in e1])
    m = ov.min()
    idx = [i * len(routes2) + j for i in range(len(routes1)) for j in range(len(routes2))
           if ov[i, j] == m]
    return float(M[idx].mean(axis=0).max()), len(idx), int(m)


def best_m_pairings(M, m_max=4):
    """In-sample cap on 'uniform over m hand-picked joint plans': m=2 exhaustive, m>2 greedy
    continuation from each of the top-40 pairs (near-optimal, fast)."""
    P = M.shape[0]
    out = {}
    v1 = M.max(axis=1)
    out[1] = float(v1.min())
    # m = 2 exhaustive
    best2, arg2 = np.inf, None
    order = np.argsort(v1)[:min(P, 120)]
    for a_i, a in enumerate(order):
        rows = (M[a] + M[order]) / 2.0
        v = rows.max(axis=1)
        v[a_i] = np.inf
        b = int(v.argmin())
        if v[b] < best2:
            best2, arg2 = float(v[b]), (int(a), int(order[b]))
    out[2] = best2
    # m = 3, 4 greedy from top-40 singles
    for m in (3, 4):
        best = np.inf
        for s0 in order[:40]:
            T = [int(s0)]
            acc = M[s0].copy()
            while len(T) < m:
                cand_vals = ((acc[None, :] * len(T) + M) / (len(T) + 1)).max(axis=1)
                cand_vals[T] = np.inf
                c = int(cand_vals.argmin())
                T.append(c)
                acc = (acc * (len(T) - 1) + M[c]) / len(T)
            best = min(best, float(acc.max()))
        out[m] = best
    return out


def main():
    G = build_graph()
    deg3 = [n for n, d in G.degree() if d >= 3]
    rng = random.Random(11)
    rows, tried = [], 0
    while len(rows) < 15 and tried < 4000:
        tried += 1
        s, t1, t2 = rng.sample(deg3, 3)
        try:
            r1 = build_route_set(G, s, t1, KX, "w")
            r2 = build_route_set(G, s, t2, KX, "w")
            if not (6 <= len(r1) <= 14 and 6 <= len(r2) <= 14):
                continue
            c1 = set().union(*(edges_of_route(r) for r in r1))
            c2 = set().union(*(edges_of_route(r) for r in r2))
            jac = len(c1 & c2) / len(c1 | c2)
            if jac < 0.05:
                continue
            M, R1, R2 = payoff_tensor(G, r1, r2)
            v_joint, _ = _row_minimiser(M)
            v_ind = best_product(M, R1, R2)
            v_dec, n_dec, min_ov = deconflict_uniform(M, r1, r2)
            caps = best_m_pairings(M)
            v_cap = min(caps.values())
            rows.append({"s": s, "t1": t1, "t2": t2, "jaccard": round(jac, 3),
                         "v_joint": v_joint, "v_indep": v_ind,
                         "v_deconflict_uniform": v_dec, "n_deconflict_pairs": n_dec,
                         "min_overlap": min_ov, "v_best_m_pairings": caps,
                         "gap_vs_indep": (v_ind - v_joint) / max(v_joint, 1e-9),
                         "gap_vs_napkin": (min(v_dec, v_ind) - v_joint) / max(v_joint, 1e-9),
                         "gap_vs_cap": (v_cap - v_joint) / max(v_joint, 1e-9)})
            print(f"({s}->{t1},{t2}) jac {jac:.2f}: joint {v_joint:.3f} | indep {v_ind:.3f} "
                  f"| deconflict-unif {v_dec:.3f} (n={n_dec}, ov={min_ov}) "
                  f"| best-m caps {'/'.join(f'{caps[m]:.3f}' for m in (1, 2, 3, 4))}", flush=True)
        except Exception:
            continue

    gi = [r["gap_vs_indep"] for r in rows]
    gn = [r["gap_vs_napkin"] for r in rows]
    gc = [r["gap_vs_cap"] for r in rows]
    print(f"\n{len(rows)} triples | median gap vs INDEPENDENT {100*np.median(gi):.1f}% "
          f"(B4's row) | vs payoff-blind NAPKIN-COORDINATED {100*np.median(gn):.1f}% "
          f"| vs in-sample m-pairing CAP {100*np.median(gc):.1f}%")
    json.dump({"rows": rows,
               "median_gap_vs_indep": float(np.median(gi)),
               "median_gap_vs_napkin": float(np.median(gn)),
               "median_gap_vs_cap": float(np.median(gc))},
              open("models/runs/b4_joint_napkin_probe.json", "w"), indent=2)
    print("[written] models/runs/b4_joint_napkin_probe.json")


if __name__ == "__main__":
    main()
