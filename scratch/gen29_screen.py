#!/usr/bin/env python3
"""gen29 screen (ORACLE-ONLY, free): the R0-style aiming screen for the three-stream
coordination act. Samples valid (s, t1, t2, t3) triples, computes the COMPLETE baseline family
(GEN29_MULTIOD_HANDOFF.md §2) per cell, and shortlists by GAP-VS-CAP (the in-sample m-pairing
cap: the hardest, oracle-fitted rule) subject to non-degeneracy (eq in (0.05, 0.9); non-uniform
joint equilibrium so smooth FP has a gradient). Emits the prevalence figure (anti-cherry-pick),
the headline cell + peers, and the train/held-out/validation split.
"""
from __future__ import annotations

import itertools
import json
import random

import networkx as nx
import numpy as np

from scratch.b4_joint_napkin_probe import best_m_pairings
from scratch.b4_multiod_probe import build_graph
from scratch.b4_widen_probe import (best_product_general, deconflict_uniform_general,
                                    joint_payoff, survival_mats)
from src.baselines.interdiction_oracle import build_route_set, edges_of_route
from src.baselines.multiconvoy_oracle import _row_minimiser

KX = 8


def disjoint_stack_product(G, rsets, S_list, isets):
    """Per-stream disjoint-stack (uniform over each stream's edge-disjoint routes), independent
    product across streams -> joint distribution -> exploitability (the R0a heuristic, composed)."""
    per = []
    for f, rs in enumerate(rsets):
        s, t = str(rs[0][0]), str(rs[0][-1])
        m = sum(1 for _ in nx.edge_disjoint_paths(G, s, t))              # disjoint prefix count
        d = np.zeros(len(rs)); d[:max(1, min(m, len(rs)))] = 1.0; d /= d.sum()
        per.append(d)
    # joint product distribution over route tuples (stream 0 outermost)
    dist = per[0]
    for nxt in per[1:]:
        dist = np.outer(dist, nxt).reshape(-1)
    M, _ = joint_payoff(S_list, isets)
    return float((dist @ M).max())


def tabular_fp(M, rounds=2500, eta=0.3):
    """Tabular smooth FP over joint plans vs the exact joint BR (ties the equilibrium; the
    'best-response-oracle methods' row)."""
    n = M.shape[0]; x = np.full(n, 1.0 / n); tot = np.zeros(n)
    for t in range(1, rounds + 1):
        tot += x
        j = int((tot / t @ M).argmax())
        x = x * np.exp(-eta * M[:, j]); x /= x.sum()
    return float((tot / rounds @ M).max())


def ent_frac(x):
    p = x[x > 1e-12]
    return float(-(p * np.log(p)).sum() / np.log(len(x))) if len(x) > 1 else 0.0


def cell(G, s, targets, K=1, want_fp=False):
    rsets = [build_route_set(G, s, t, KX, "w") for t in targets]
    if any(not (4 <= len(r) <= 14) for r in rsets):
        return None
    # pairwise corridor sharing
    ce = [set().union(*(edges_of_route(r) for r in rs)) for rs in rsets]
    jac = min(len(ce[a] & ce[b]) / len(ce[a] | ce[b])
              for a in range(len(ce)) for b in range(a + 1, len(ce)))
    if jac < 0.05:
        return None
    S_list, cand = survival_mats(G, rsets)
    E = len(cand)
    isets = [(e,) for e in range(E)] if K == 1 else list(itertools.combinations(range(E), K))
    M, shape = joint_payoff(S_list, isets)
    if M.shape[0] * M.shape[1] > 8_000_000:
        return None
    v_joint, dstar = _row_minimiser(M)
    if not (0.05 < v_joint < 0.9):
        return None
    v_det = float(M.max(axis=1).min())
    v_ind = best_product_general(M, shape)
    v_dec, _, _ = deconflict_uniform_general(M, rsets)
    v_dis = disjoint_stack_product(G, rsets, S_list, isets)
    v_cap = min(best_m_pairings(M).values())
    napkin = min(v_ind, v_dec, v_dis)
    fp = tabular_fp(M) if want_fp else None
    return dict(s=s, targets=list(targets), K=K, F=len(targets), E=E, R=shape, jac=round(jac, 3),
                eq=v_joint, det=v_det, indep=v_ind, deconflict=v_dec, disjoint=v_dis,
                cap=v_cap, tabular_fp=fp, ent_frac=ent_frac(dstar),
                gap_vs_cap=(v_cap - v_joint) / v_joint,
                gap_vs_napkin=(napkin - v_joint) / v_joint)


def main():
    G = build_graph()
    deg3 = [n for n, d in G.degree() if d >= 3]
    rng = random.Random(29)
    # seed with the known-good triples from the widen probe, then sample
    known = [("147", "212", "188", "195"), ("147", "212", "188", "115"),
             ("147", "212", "188", "127"), ("119", "62", "278", "59"),
             ("119", "62", "278", "0"), ("119", "62", "278", "181")]
    cells, seen = [], set()
    for s, t1, t2, t3 in known:
        c = cell(G, s, (t1, t2, t3))
        if c:
            cells.append(c); seen.add((s, t1, t2, t3))
            print(f"{s}->{','.join((t1,t2,t3)):14s} eq={c['eq']:.3f} cap={c['cap']:.3f} "
                  f"gap-vs-cap {100*c['gap_vs_cap']:.0f}% H={c['ent_frac']:.2f}", flush=True)
    tried = 0
    while len(cells) < 55 and tried < 6000:
        tried += 1
        s, t1, t2, t3 = rng.sample(deg3, 4)
        key = (s, t1, t2, t3)
        if key in seen:
            continue
        seen.add(key)
        try:
            c = cell(G, s, (t1, t2, t3))
        except Exception:
            continue
        if c:
            cells.append(c)
            if len(cells) % 5 == 0:
                print(f"  ...{len(cells)} cells (last {s}->{t1},{t2},{t3} "
                      f"gap-vs-cap {100*c['gap_vs_cap']:.0f}%)", flush=True)

    # shortlist: non-degenerate + gradient, ranked by gap-vs-cap
    ok = [c for c in cells if c["ent_frac"] < 0.92 and c["gap_vs_cap"] > 0.1]
    ok.sort(key=lambda c: -c["gap_vs_cap"])
    print(f"\n=== {len(cells)} valid cells; {len(ok)} pass non-degeneracy + gap>10% ===")
    for c in ok[:12]:
        print(f"{c['s']}->{','.join(c['targets']):14s} eq={c['eq']:.3f} cap={c['cap']:.3f} "
              f"indep={c['indep']:.3f} det={c['det']:.3f} | gap-vs-cap {100*c['gap_vs_cap']:.0f}% "
              f"vs-napkin {100*c['gap_vs_napkin']:.0f}% H={c['ent_frac']:.2f}")
    caps = np.array([c["gap_vs_cap"] for c in cells])
    print(f"\nprevalence: median gap-vs-cap {100*np.median(caps):.0f}%, "
          f">20% on {int((caps>0.2).sum())}/{len(cells)}, >35% on {int((caps>0.35).sum())}/{len(cells)}")
    # HEADLINE = the handoff's pre-registered cell (147->212,188,195): cap strictly < det (a
    # genuine mixture cap, not collapsed to determinism), so 'beat the cap' is a real
    # class-separation claim. Chosen before any training, per the discipline; auto-top cells with
    # cap==det are weaker headlines and stay in the pool only.
    HEAD_KEY = ("147", ["212", "188", "195"])
    head = next((c for c in ok if (c["s"], c["targets"]) == HEAD_KEY), ok[0])
    rest = [c for c in ok if c is not head]
    # prefer genuine-cap cells (cap < det - 0.03) for the gated held-out set (clean bars)
    clean = [c for c in rest if c["cap"] < c["det"] - 0.03]
    other = [c for c in rest if c not in clean]
    pool = (clean[:15] if len(clean) >= 15 else clean + other)[:15]
    remaining = [c for c in rest if c not in pool]
    held = remaining[:6]; val = remaining[6:10]
    for c in [head]:
        M0, _ = joint_payoff(survival_mats(G, [build_route_set(G, c['s'], t, KX, 'w')
                             for t in c['targets']])[0], [(e,) for e in range(c['E'])])
        c["tabular_fp"] = tabular_fp(M0)
    split = {"headline": head, "pool": pool, "held_out": held, "validation": val,
             "all_cells": cells}
    json.dump(split, open("models/runs/gen29_screen.json", "w"), indent=1)
    print(f"\nHEADLINE {head['s']}->{','.join(head['targets'])}: eq {head['eq']:.3f} cap {head['cap']:.3f} "
          f"gap-vs-cap {100*head['gap_vs_cap']:.0f}% tabFP {head['tabular_fp']:.3f}")
    print(f"pool {len(pool)}, held-out {len(held)}, validation {len(val)} -> models/runs/gen29_screen.json")

    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(100 * caps, bins=18, color="#4a74c0", alpha=0.85, edgecolor="white")
        ax.axvline(100 * head["gap_vs_cap"], color="#c0503a", lw=2,
                   label=f"headline {100*head['gap_vs_cap']:.0f}%")
        ax.axvline(100 * np.median(caps), color="#2c8a6a", lw=1.5, ls="--",
                   label=f"median {100*np.median(caps):.0f}%")
        ax.set_xlabel("correlation gap vs the in-sample m-pairing cap (%)")
        ax.set_ylabel("cells"); ax.set_title("gen29 prevalence: multi-OD coordination gap over the population")
        ax.legend(); ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); fig.savefig("assets/gen29_prevalence.png", dpi=150)
        print("wrote assets/gen29_prevalence.png")
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
