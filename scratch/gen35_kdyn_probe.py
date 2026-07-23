#!/usr/bin/env python3
"""gen35 DESIGN PROBE (oracle-only, no training): the dynamic register at K > 1.

Question (gates the gen35 pre-registration): as the interdictor's per-sortie budget K grows
toward the number of disjoint routes m, does the gap between the best NAIVE dynamic rule
(rotation / anti-repeat, the gen27 composed family) and the exact dynamic optimum
(history_opt, RVI) WIDEN? And does within-episode dynamism still pay at all
(history_opt << iid_eq)?

The target regime for a new act is: dynamism pays AND naive dynamic rules fail. If at high K
the composed rules collapse onto iid_eq while history_opt also collapses onto iid_eq (no room
to dodge), the game is degenerate there and that is the reported scoping result.

Mechanics identical to gen19/gen27 (softmax-BR tau=0.15 to the trailing w=3 realised-route
window; fleet stacks, N=3, mission objective; band (0.15,0.95), k_extra=8, kaliningrad graph).
K enters only through the interdiction-set columns of the stacked loss matrix L; the RVI state
space stays R^w (K-agnostic), so every yardstick here is EXACT.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen35_kdyn_probe.py
Writes models/runs/gen35_kdyn_probe.json
"""
from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np
import torch

from scripts.train_b1lite1 import oracle_refs, softmax_br, stacked_L
from scratch.critique_followup_probes import (
    antirepeat_value, disjoint_subset, rotation_value, static_value)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(1)
N, BAND, KX, W, TAU = 3, (0.15, 0.95), 8, 3, 0.15
ODS = ["35-159", "62-97", "71-33"]
KS = [1, 2, 3]
MAX_ISETS = 250_000  # skip any (od, K) whose exact column count exceeds this


def inv_vuln_dist(dis, L, R):
    q = np.array([L[r].max() for r in dis])
    w_ = 1.0 / np.maximum(q, 1e-9)
    d = np.zeros(R)
    d[np.array(dis)] = w_ / w_.sum()
    return d


def uniform_dist(dis, R):
    d = np.zeros(R)
    d[np.array(dis)] = 1.0 / len(dis)
    return d


def best_rotation(dis, L):
    orders = [tuple(dis)]
    rng = np.random.default_rng(0)
    n_perm = min(20, math.factorial(len(dis)))
    seen = set(orders)
    while len(orders) < n_perm:
        p = tuple(rng.permutation(dis).tolist())
        if p not in seen:
            seen.add(p)
            orders.append(p)
    return min(rotation_value(list(o), L, TAU, W) for o in orders)


def main():
    out = {"config": dict(N=N, band=BAND, k_extra=KX, w=W, tau=TAU, ods=ODS, ks=KS)}
    rows = []
    for od in ODS:
        s, t = od.split("-")
        for K in KS + ([4] if od == "71-33" else []):
            t0 = time.time()
            try:
                env = make_multiconvoy_env(od=(s, t), N=N, K=K, k_extra_routes=KX,
                                           menu_select=True, edge_vuln_band=BAND,
                                           interception_loss=10.0)
            except Exception as e:  # noqa: BLE001 - report and continue the grid
                rows.append(dict(od=od, K=K, error=str(e)))
                continue
            game = env.game
            nj = game.payoff.shape[1]
            if nj > MAX_ISETS:
                rows.append(dict(od=od, K=K, skipped=f"n_isets {nj} > {MAX_ISETS}"))
                continue
            L = stacked_L(game, N)
            R = L.shape[0]
            dis = disjoint_subset([set(e) for e in game.route_edges])
            m = len(dis)
            refs = oracle_refs(L, TAU, W)
            rot = best_rotation(dis, L)
            anti_dis = antirepeat_value(dis, L, TAU, W)
            anti_full = antirepeat_value(list(range(R)), L, TAU, W)
            st_uni = static_value(uniform_dist(dis, R), L, TAU, W)
            st_inv = static_value(inv_vuln_dist(dis, L, R), L, TAU, W)
            naive_dyn = dict(rotation=rot, anti_disjoint=anti_dis, anti_full=anti_full)
            best_name = min(naive_dyn, key=naive_dyn.get)
            row = dict(od=od, K=K, R=R, n_isets=nj, m_disjoint=m,
                       v_eq_oneshot=refs["v_eq"], iid_eq=refs["iid_eq"],
                       static_det=refs["static_det"], history_opt=refs["history_opt"],
                       rotation=rot, anti_disjoint=anti_dis, anti_full=anti_full,
                       static_uniform_disjoint=st_uni, static_invvuln_disjoint=st_inv,
                       best_naive_dynamic=naive_dyn[best_name], best_naive_name=best_name,
                       dyn_gain_iid_over_hist=refs["iid_eq"] / max(refs["history_opt"], 1e-12),
                       naive_over_hist=naive_dyn[best_name] / max(refs["history_opt"], 1e-12),
                       naive_over_iid=naive_dyn[best_name] / max(refs["iid_eq"], 1e-12),
                       secs=round(time.time() - t0, 1))
            rows.append(row)
            print(f"{od} K={K} R={R} nj={nj} m={m} | eq1shot {refs['v_eq']:.4f} "
                  f"iid_eq {refs['iid_eq']:.4f} det {refs['static_det']:.4f} "
                  f"hist_opt {refs['history_opt']:.4f} | rot {rot:.4f} antiD {anti_dis:.4f} "
                  f"antiF {anti_full:.4f} | naive/hist {row['naive_over_hist']:.3f} "
                  f"iid/hist {row['dyn_gain_iid_over_hist']:.3f} ({row['secs']}s)", flush=True)
    out["rows"] = rows
    with open("models/runs/gen35_kdyn_probe.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote models/runs/gen35_kdyn_probe.json")


if __name__ == "__main__":
    main()
