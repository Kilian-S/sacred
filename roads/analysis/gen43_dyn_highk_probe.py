#!/usr/bin/env python3
"""gen43 dynamic high-K probe (oracle/eval-only, no training).

Extends the dynamic arm to K=5 and K=6 on 71-33 (m=6, R=11, kx=8, N=3, band 0.15-0.95, w=3,
tau=0.15) by solving the exact softmax game directly. For each K, builds the closed-form loss
matrix L = 1 - (1 - payoff)^N (verified against the trainer's stacked_L at K=4) and the
window-MDP cost matrix, then computes the exact optimum (Karp minimum mean cycle, cross-checked
by damped relative value iteration), best-of-20-orders disjoint rotation, a composed anti-repeat
policy, and the one-shot iid equilibrium value, with wall-clock timing throughout. K=7 and K=8
costs are extrapolated from the measured per-column cost.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    analysis/gen43_dyn_highk_probe.py
Artefact: models/runs/gen43_dyn_highk_probe.json
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from analysis.dyn_exact import build_window_mdp, damped_rvi, karp_mmc
from scripts.train_b1lite1 import stacked_L
from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, BAND, KX, W, TAU = 3, (0.15, 0.95), 8, 3, 0.15
OUT: dict = {}


def disjoint_subset(route_edges):
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (re_ & used):
            kept.append(i)
            used |= re_
    return kept


def vec_L(game):
    return 1.0 - (1.0 - game.payoff) ** N


def sid(win, R, w):
    x = 0
    for i in win:
        x = x * R + i
    return x


def rotation_best(cost, dis, R, w, orders=20):
    rng = np.random.default_rng(0)
    perms = [list(dis)] + [list(rng.permutation(dis)) for _ in range(orders - 1)]
    best = np.inf
    for seq in perms:
        m = len(seq)
        v = 0.0
        for t in range(m):
            win = tuple(seq[(t - 1 - i) % m] for i in range(w))[::-1]
            v += cost[sid(win, R, w), seq[t]]
        best = min(best, v / m)
    return float(best)


def antirepeat_core(cost, dis, R, w, iters=800):
    states = list(itertools.product(dis, repeat=w))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    c = np.zeros(n)
    for s in states:
        i = idx[s]
        allowed = [r for r in dis if r not in s] or list(dis)
        c[i] = float(np.mean([cost[sid(s, R, w), r] for r in allowed]))
        for r in allowed:
            P[i, idx[s[1:] + (r,)]] += 1.0 / len(allowed)
    pi = np.ones(n) / n
    for _ in range(iters):
        pi = pi @ P
    return float(pi @ c)


def iid_value(cost, d, R, w):
    v = 0.0
    for win in itertools.product(range(R), repeat=w):
        pw = 1.0
        for i in win:
            pw *= d[i]
        if pw > 0:
            v += pw * float(d @ cost[sid(win, R, w)])
    return v


def main():
    # closed-form L verified against the trainer's stacked_L at K=4 first
    env4 = make_multiconvoy_env(("71", "33"), N=N, K=4, k_extra_routes=KX,
                                edge_vuln_band=BAND, absolute_vuln_norm=True,
                                menu_select=True, objective="mission")
    diff = float(np.abs(vec_L(env4.game) - stacked_L(env4.game, N)).max())
    print(f"closed-form L vs trainer stacked_L at K=4: max abs diff {diff:.2e}", flush=True)
    assert diff < 1e-12
    OUT["L_closed_form_check_K4"] = diff

    for K in (5, 6):
        t0 = time.time()
        env = make_multiconvoy_env(("71", "33"), N=N, K=K, k_extra_routes=KX,
                                   edge_vuln_band=BAND, absolute_vuln_norm=True,
                                   menu_select=True, objective="mission")
        t_env = time.time() - t0
        game = env.game
        n_isets = game.payoff.shape[1]
        t0 = time.time()
        L = vec_L(game)
        t_L = time.time() - t0
        R = L.shape[0]
        dis = disjoint_subset(game.route_edges)
        t0 = time.time()
        cost, n, R_, pw = build_window_mdp(L, TAU, W)
        t_cost = time.time() - t0
        t0 = time.time()
        opt = karp_mmc(cost, n, R_, pw)
        t_karp = time.time() - t0
        rvi, rvi_conv = damped_rvi(cost, n, R_, pw)
        rot = rotation_best(cost, dis, R, W)
        anti = antirepeat_core(cost, dis, R, W)
        t0 = time.time()
        v_eq, eq = _row_minimiser(L)
        t_lp = time.time() - t0
        iid = iid_value(cost, eq, R, W)
        det = min(float(cost[sid((r,) * W, R, W), r]) for r in range(R))
        best_rule = min(rot, anti)
        row = {"n_isets": int(n_isets), "opt_karp": round(float(opt), 4),
               "opt_rvi_check": round(float(rvi), 4), "rvi_converged": bool(rvi_conv),
               "rotation_best20": round(rot, 4), "antirepeat_core": round(anti, 4),
               "iid_eq": round(iid, 4), "static_det": round(det, 4),
               "v_eq_oneshot": round(float(v_eq), 4),
               "rule_over_opt": round(best_rule / opt, 3),
               "iid_over_opt": round(iid / opt, 3),
               "secs": {"env": round(t_env, 1), "L": round(t_L, 1),
                        "cost_matrix": round(t_cost, 1), "karp": round(t_karp, 1),
                        "lp": round(t_lp, 1)},
               "per_softmax_ms": round(1000 * t_cost / n, 2)}
        OUT[f"K{K}"] = row
        print(f"K={K}: {json.dumps(row)}", flush=True)

    # K=7/8 extrapolation from measured per-column cost
    ms6 = OUT["K6"]["per_softmax_ms"] / OUT["K6"]["n_isets"]
    for K, ni in ((7, 32224114), (8, 141120525)):
        per_call = ms6 * ni
        OUT[f"K{K}_extrapolated"] = {
            "n_isets": ni, "L_bytes_float64_GB": round(11 * ni * 8 / 1e9, 1),
            "per_softmax_ms": round(per_call, 1),
            "cost_matrix_min": round(1331 * per_call / 60000, 1),
            "train_softmax_hours_8000_sorties": round(8000 * per_call / 3.6e6, 2),
            "eval2000_min_per_eval": round(2000 * per_call / 60000, 1)}
        print(f"K={K} extrapolated: {json.dumps(OUT[f'K{K}_extrapolated'])}", flush=True)

    json.dump(OUT, open("models/runs/gen43_dyn_highk_probe.json", "w"), indent=2)
    print("[written] models/runs/gen43_dyn_highk_probe.json", flush=True)


if __name__ == "__main__":
    main()
