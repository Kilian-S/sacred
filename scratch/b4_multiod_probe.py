#!/usr/bin/env python3
"""B4 (ORACLE-ONLY): the multi-OD correlation-gap probe.

N=2 convoys with DIFFERENT destinations sharing corridor edges (Kaliningrad triples s -> t1, t2).
v_joint = exact minimax over JOINT route pairs (LP over R1 x R2 vs K=1 isets of the union of
candidate edges); v_indep = best PRODUCT distribution (alternating per-convoy LP best responses,
5 restarts; a local optimum = an upper bound on the true independent value, disclosed);
v_det = best deterministic pair. Mission objective P(>=1 lost), independence given the iset.
Pre-registration: experiments/b3_b4_oracle.md §B4 (median gap >= 10% justifies the Tier-3 game).
"""
from __future__ import annotations

import itertools
import json
import random

import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import (
    build_route_set, edges_of_route, length_band_vulnerability)
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import _DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS
from src.utils.graph_utils import load_osm_graph_and_demands

BAND, KX = (0.15, 0.95), 8


def build_graph():
    nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    return G.subgraph(max(nx.connected_components(G), key=len)).copy()


def payoff_tensor(G, routes1, routes2):
    """M[(i,j), e] = P(>=1 of the two convoys lost) if edge e is interdicted."""
    e1 = [edges_of_route(r) for r in routes1]
    e2 = [edges_of_route(r) for r in routes2]
    cand = sorted(set().union(*e1, *e2), key=lambda e: tuple(sorted(map(str, e))))
    vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=list(G.edges()))
    R1, R2, E = len(routes1), len(routes2), len(cand)
    P1 = np.zeros((R1, E))
    P2 = np.zeros((R2, E))
    for k, e in enumerate(cand):
        for i, es in enumerate(e1):
            P1[i, k] = vuln[e] if e in es else 0.0
        for j, es in enumerate(e2):
            P2[j, k] = vuln[e] if e in es else 0.0
    # mission: 1 - (1-p1)(1-p2)
    M = np.zeros((R1 * R2, E))
    for i in range(R1):
        for j in range(R2):
            M[i * R2 + j] = 1.0 - (1.0 - P1[i]) * (1.0 - P2[j])
    return M, R1, R2


def best_product(M, R1, R2, restarts=5, iters=60, seed=0):
    """Alternating per-convoy LP best responses on the product family; returns the best local
    value found (an UPPER bound on the true independent optimum)."""
    E = M.shape[1]
    Mt = M.reshape(R1, R2, E)
    best = np.inf
    rng = np.random.default_rng(seed)
    for r in range(restarts):
        x2 = rng.dirichlet(np.ones(R2)) if r else np.full(R2, 1.0 / R2)
        val = np.inf
        for _ in range(iters):
            M1 = np.einsum("j,ije->ie", x2, Mt)      # convoy-1 payoff given x2
            v1, x1 = _row_minimiser(M1)
            M2 = np.einsum("i,ije->je", x1, Mt)
            v2, x2 = _row_minimiser(M2)
            if abs(v2 - val) < 1e-9:
                break
            val = v2
        best = min(best, val)
    return float(best)


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
                continue                       # corridor-sharing triples only
            M, R1, R2 = payoff_tensor(G, r1, r2)
            v_joint, _ = _row_minimiser(M)
            v_det = float(M.max(axis=1).min())
            v_ind = best_product(M, R1, R2)
            gap = (v_ind - v_joint) / max(v_joint, 1e-9)
            rows.append({"s": s, "t1": t1, "t2": t2, "jaccard": round(jac, 3),
                         "v_det": v_det, "v_indep": v_ind, "v_joint": v_joint,
                         "corr_gap": gap})
            print(f"({s}->{t1},{t2}) jac {jac:.2f}: det {v_det:.3f} indep {v_ind:.3f} "
                  f"joint {v_joint:.3f} | corr gap {100*gap:.1f}%", flush=True)
        except Exception:
            continue

    gaps = [r["corr_gap"] for r in rows]
    print(f"\nB4: {len(rows)} corridor-sharing triples | median corr gap {100*np.median(gaps):.1f}% "
          f"| >=10% on {sum(g >= 0.10 for g in gaps)}/{len(rows)}")
    json.dump({"rows": rows, "median_gap": float(np.median(gaps))},
              open("models/runs/b4_multiod_probe.json", "w"), indent=2)
    print("[written] models/runs/b4_multiod_probe.json")


if __name__ == "__main__":
    main()
