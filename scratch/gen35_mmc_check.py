#!/usr/bin/env python3
"""YARDSTICK VERIFICATION (oracle-only): is oracle_refs' history_opt (RVI) the true dynamic
optimum?

Trigger: scratch/gen35_kdyn_probe.py measured deterministic ROTATION at 0.0413 on 35-159 K=1,
BELOW the RVI history_opt 0.0488. Rotation is itself a stationary policy on the R^w window MDP,
so it cannot beat the true optimum: either the RVI failed to converge (deterministic-transition
periodicity) or rotation_value is inconsistent with the MDP cost convention.

Independent exact method: the window MDP has DETERMINISTIC transitions, so its optimal average
cost is exactly the MINIMUM MEAN CYCLE of the (state --action--> state) graph. Karp's algorithm
computes it exactly, no iteration/tolerance. Also cross-checks a long damped RVI.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen35_mmc_check.py
Writes models/runs/gen35_mmc_check.json
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch

from scripts.train_b1lite1 import oracle_refs, softmax_br, stacked_L
from scratch.critique_followup_probes import disjoint_subset, rotation_value
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(1)
N, BAND, KX, W, TAU = 3, (0.15, 0.95), 8, 3, 0.15
CELLS = [("35-159", 1), ("35-159", 2), ("35-159", 3),
         ("62-97", 1), ("62-97", 2), ("62-97", 3),
         ("71-33", 1), ("71-33", 2), ("71-33", 3), ("71-33", 4)]


def build_mdp(L, tau=TAU, w=W):
    """cost[s,a], nxt[s,a] for the window MDP (identical convention to oracle_refs)."""
    R = L.shape[0]
    states = list(itertools.product(range(R), repeat=w))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    cost = np.zeros((n, R))
    nxt = np.zeros((n, R), dtype=np.int64)
    for si, s in enumerate(states):
        br = softmax_br(np.bincount(s, minlength=R).astype(float), L, tau)
        cost[si] = L @ br
        for a in range(R):
            nxt[si, a] = idx[s[1:] + (a,)]
    return cost, nxt, n, R


def karp_min_mean_cycle(cost, nxt, n, R):
    """Exact minimum mean cycle via Karp. Edges: (s -> nxt[s,a], cost[s,a])."""
    src = np.repeat(np.arange(n), R)
    dst = nxt.reshape(-1)
    w_ = cost.reshape(-1)
    d = np.full((n + 1, n), np.inf)
    d[0] = 0.0
    for k in range(1, n + 1):
        cand = d[k - 1][src] + w_
        row = np.full(n, np.inf)
        np.minimum.at(row, dst, cand)
        d[k] = row
    ks = np.arange(n)[:, None]  # k = 0..n-1
    with np.errstate(invalid="ignore"):
        ratios = (d[n][None, :] - d[:n]) / (n - ks)
    ratios = np.where(np.isfinite(ratios), ratios, -np.inf)
    per_v = ratios.max(axis=0)
    per_v = np.where(np.isfinite(d[n]), per_v, np.inf)
    return float(per_v.min())


def damped_rvi(cost, nxt, iters=200_000, damp=0.5, tol=1e-12):
    """Damped relative value iteration; returns the gain estimate from the Bellman residual."""
    n = cost.shape[0]
    h = np.zeros(n)
    g = 0.0
    for _ in range(iters):
        q = (cost + h[nxt]).min(axis=1)
        g_new = q.mean() - h.mean()
        h_new = q - q[0]
        if np.max(np.abs(h_new - h)) < tol:
            h, g = h_new, g_new
            break
        h = damp * h_new + (1 - damp) * h
        g = g_new
    return float(g)


def main():
    rows = []
    for od, K in CELLS:
        s, t = od.split("-")
        t0 = time.time()
        env = make_multiconvoy_env(od=(s, t), N=N, K=K, k_extra_routes=KX, menu_select=True,
                                   edge_vuln_band=BAND, interception_loss=10.0)
        L = stacked_L(env.game, N)
        cost, nxt, n, R = build_mdp(L)
        mmc = karp_min_mean_cycle(cost, nxt, n, R)
        rvi_orig = oracle_refs(L, TAU, W)["history_opt"]
        rvi_damp = damped_rvi(cost, nxt)
        dis = disjoint_subset([set(e) for e in env.game.route_edges])
        rot = rotation_value(list(dis), L, TAU, W)
        rows.append(dict(od=od, K=K, n_states=n, R=R,
                         mmc_exact=mmc, rvi_original=rvi_orig, rvi_damped=rvi_damp,
                         rotation=rot, rvi_over_mmc=rvi_orig / mmc if mmc > 0 else None,
                         secs=round(time.time() - t0, 1)))
        print(f"{od} K={K}: mmc_EXACT {mmc:.5f} | rvi_orig {rvi_orig:.5f} "
              f"rvi_damped {rvi_damp:.5f} | rotation {rot:.5f} | "
              f"rvi/mmc {rvi_orig/mmc:.3f} ({rows[-1]['secs']}s)", flush=True)
    with open("models/runs/gen35_mmc_check.json", "w") as f:
        json.dump(dict(config=dict(N=N, band=BAND, k_extra=KX, w=W, tau=TAU), rows=rows), f, indent=1)
    print("wrote models/runs/gen35_mmc_check.json")


if __name__ == "__main__":
    main()
