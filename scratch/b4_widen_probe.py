#!/usr/bin/env python3
"""B4 widening probe (ORACLE-ONLY, 2026-07-18, Critic Aerial): how large can the multi-OD
correlation gap get, honestly?

Axes measured on the screened-best triples from b4_joint_napkin_probe.json:
  1. K=2 (the attacker covers two edges: coordination must dodge PAIRS of corridors);
  2. F=3 (three supply streams from one base: the joint object is triples);
both scored against the SAME complete baseline family as the napkin probe: best independent
product (alternating LPs, restarts), payoff-blind deconflict-uniform, and the IN-SAMPLE cap
(best uniform mixture over <= 4 oracle-picked joint plans). Every gap is reported vs the CAP
(the hardest row), so nothing here can be a strawman. Loss-averse mission objective throughout
(the additive objective provably has NO correlation gap: value is linear in the marginals, so
the mission coupling is load-bearing for coordination, the B3 law extended).
"""
from __future__ import annotations

import itertools
import json
import random

import numpy as np

from scratch.b4_multiod_probe import build_graph, payoff_tensor
from scratch.b4_joint_napkin_probe import best_m_pairings
from src.baselines.interdiction_oracle import (
    build_route_set, edges_of_route, length_band_vulnerability)
from src.baselines.multiconvoy_oracle import _row_minimiser

BAND, KX = (0.15, 0.95), 8


def survival_mats(G, route_sets):
    """Per-flow single-edge SURVIVAL matrices on the shared candidate edge list."""
    e_sets = [[edges_of_route(r) for r in rs] for rs in route_sets]
    cand = sorted(set().union(*(es for flow in e_sets for es in flow)),
                  key=lambda e: tuple(sorted(map(str, e))))
    vuln = length_band_vulnerability(G, cand, band=BAND, weight="w", norm_edges=list(G.edges()))
    mats = []
    for flow in e_sets:
        P = np.zeros((len(flow), len(cand)))
        for k, e in enumerate(cand):
            for i, es in enumerate(flow):
                P[i, k] = vuln[e] if e in es else 0.0
        mats.append(1.0 - P)                      # survival vs a single interdicted edge
    return mats, cand


def joint_payoff(S_list, isets):
    """M[joint_route_tuple, iset] = 1 - prod_f prod_{e in iset} S_f[route_f, e] (mission)."""
    per_flow = []
    for S in S_list:
        logS = np.log(np.clip(S, 1e-300, 1.0))
        per_flow.append(np.exp(logS[:, np.asarray(isets)].sum(axis=2)))   # [R_f, n_isets]
    shape = [S.shape[0] for S in S_list]
    surv = per_flow[0]
    for nxt in per_flow[1:]:
        surv = (surv[:, None, :] * nxt[None, :, :]).reshape(-1, surv.shape[-1])
    return 1.0 - surv, shape


def best_product_general(M, shape, restarts=4, iters=50, seed=0):
    """Alternating per-flow LP best responses over the product family (upper bound on the
    independent optimum, disclosed)."""
    F = len(shape)
    Mt = M.reshape(*shape, M.shape[1])
    rng = np.random.default_rng(seed)
    best = np.inf
    for r in range(restarts):
        xs = [np.full(n, 1.0 / n) if r == 0 else rng.dirichlet(np.ones(n)) for n in shape]
        val = np.inf
        for _ in range(iters):
            for f in range(F):
                T = Mt
                for g in range(F):
                    if g == f:
                        continue
                    ax = g if g < f else 1   # after earlier contractions axes shift
                T = Mt
                # contract all flows except f, in order
                for g in reversed(range(F)):
                    if g == f:
                        continue
                    T = np.tensordot(xs[g], T, axes=([0], [g if g < f else g]))
                v, xs[f] = _row_minimiser(T)
            if abs(v - val) < 1e-9:
                break
            val = v
        best = min(best, val)
    return float(best)


def deconflict_uniform_general(M, route_sets):
    e_sets = [[edges_of_route(r) for r in rs] for rs in route_sets]
    combos = list(itertools.product(*[range(len(rs)) for rs in route_sets]))
    ov = np.array([sum(len(e_sets[a][c[a]] & e_sets[b][c[b]])
                       for a in range(len(c)) for b in range(a + 1, len(c)))
                   for c in combos])
    m = ov.min()
    idx = np.where(ov == m)[0]
    return float(M[idx].mean(axis=0).max()), int(len(idx)), int(m)


def cell(tag, G, s, targets, K):
    rsets = [build_route_set(G, s, t, KX, "w") for t in targets]
    if any(not (4 <= len(r) <= 14) for r in rsets):
        return None
    S_list, cand = survival_mats(G, rsets)
    E = len(cand)
    isets = ([ (e,) for e in range(E) ] if K == 1
             else list(itertools.combinations(range(E), 2)))
    M, shape = joint_payoff(S_list, isets)
    if M.shape[0] * M.shape[1] > 80_000_000:
        return None
    v_joint, _ = _row_minimiser(M)
    v_ind = best_product_general(M, shape)
    v_dec, n_dec, ovl = deconflict_uniform_general(M, rsets)
    caps = best_m_pairings(M)
    v_cap = min(caps.values())
    napkin = min(v_ind, v_dec)
    print(f"{tag:26s} F={len(targets)} K={K} E={E} joint={v_joint:.3f} | indep={v_ind:.3f} "
          f"deconf={v_dec:.3f} cap={v_cap:.3f} | GAP vs-napkin {100*(napkin-v_joint)/v_joint:.0f}% "
          f"vs-CAP {100*(v_cap-v_joint)/v_joint:.0f}%", flush=True)
    return {"tag": tag, "F": len(targets), "K": K, "E": E, "v_joint": v_joint,
            "v_indep": v_ind, "v_deconflict": v_dec, "v_cap": v_cap,
            "gap_vs_napkin": (napkin - v_joint) / v_joint,
            "gap_vs_cap": (v_cap - v_joint) / v_joint}


def main():
    G = build_graph()
    out = []
    # the two screened-best K=1 pairs, re-derived, then their K=2 rows
    for s, t1, t2 in (("147", "212", "188"), ("119", "62", "278")):
        out.append(cell(f"{s}->{t1},{t2}", G, s, (t1, t2), K=1))
        out.append(cell(f"{s}->{t1},{t2}", G, s, (t1, t2), K=2))
    # three-stream cells: extend the best pairs with a third screened target
    rng = random.Random(7)
    deg3 = [n for n, d in G.degree() if d >= 3]
    for s, t1, t2 in (("147", "212", "188"), ("119", "62", "278")):
        added = 0
        for t3 in rng.sample(deg3, len(deg3)):
            if t3 in (s, t1, t2):
                continue
            try:
                r = cell(f"{s}->{t1},{t2},{t3}", G, s, (t1, t2, t3), K=1)
            except Exception:
                continue
            if r is not None:
                out.append(r)
                added += 1
            if added >= 3:
                break
    out = [r for r in out if r]
    json.dump(out, open("models/runs/b4_widen_probe.json", "w"), indent=2)
    print("[written] models/runs/b4_widen_probe.json")


if __name__ == "__main__":
    main()
