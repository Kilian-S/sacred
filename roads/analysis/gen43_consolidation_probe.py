#!/usr/bin/env python3
"""gen43 consolidation probe (oracle/eval-only, no training).

Checks the unified Act-2 K-ladder on 71-33 (m=6, R=11, kx=8, N=3, band 0.15-0.95, mission
objective): (S) greedy-yardstick values of naive stacks, static_det, and tabular smooth FP
at K=5..10; (X) exact equilibrium value, exact stack values, and greedy-vs-exact fidelity at
K=1..4; (D) a dynamic K=4 cost model that times stacked_L, softmax_br, and an eval-style loop
to project the 8000-sortie 3-seed training cost.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     analysis/gen43_consolidation_probe.py
Artefact: models/runs/gen43_consolidation_probe.json (regenerable, fully deterministic).
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np

from scripts.train_b1lite1 import oracle_refs, softmax_br, stacked_L
from src.baselines.multiconvoy_oracle import (
    _row_minimiser, greedy_br_attacker, objective_matrix)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, BAND, KX = 3, (0.15, 0.95), 8
OUT: dict = {}


def disjoint_subset(route_edges):
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (re_ & used):
            kept.append(i)
            used |= re_
    return kept


def stack_support(weights_by_route, R):
    tot = sum(weights_by_route.values())
    return [(tuple(N if i == r else 0 for i in range(R)), w / tot)
            for r, w in weights_by_route.items() if w > 0]


def loss_vector(game, vuln_fs, S):
    """Mission objective for a stacked fleet under interdiction set S: 1 - (1 - q_r)^N."""
    R = game.n_routes
    out = np.zeros(R)
    for r in range(R):
        surv = 1.0
        for e in S:
            if e in game.route_edges[r]:
                surv *= 1.0 - vuln_fs[e]
        out[r] = 1.0 - surv ** N
    return out


def tabular_fp(game, vuln_fs, k, T=300, eta=0.5):
    R = game.n_routes
    d_bar = np.ones(R) / R
    logw = np.log(d_bar)
    for t in range(1, T + 1):
        S, _ = greedy_br_attacker(game.route_edges, vuln_fs,
                                  stack_support({r: d_bar[r] for r in range(R)}, R), N, k)
        logw -= eta * loss_vector(game, vuln_fs, S)
        logw -= logw.max()
        d = np.exp(logw)
        d /= d.sum()
        d_bar = (d_bar * t + d) / (t + 1)
    _, v = greedy_br_attacker(game.route_edges, vuln_fs,
                              stack_support({r: d_bar[r] for r in range(R)}, R), N, k)
    return float(v), d_bar


def part_s(game, vuln_fs):
    print("=== S. static saturation, greedy yardstick, 71-33 kx=8, K=5..10 ===", flush=True)
    R = game.n_routes
    dis = disjoint_subset(game.route_edges)
    q = {r: 1.0 - (1.0 - float(game.payoff[r].max())) ** N for r in range(R)}
    arms = {
        "uniform_disjoint": {r: 1.0 for r in dis},
        "inv_vuln_disjoint": {r: 1.0 / max(q[r], 1e-9) for r in dis},
        "uniform_full": {r: 1.0 for r in range(R)},
        "inv_vuln_full": {r: 1.0 / max(q[r], 1e-9) for r in range(R)},
    }
    rows = {}
    for k in (5, 6, 7, 8, 9, 10):
        row = {}
        for name, wts in arms.items():
            _, v = greedy_br_attacker(game.route_edges, vuln_fs, stack_support(wts, R), N, k)
            row[name] = round(float(v), 4)
        # static_det: best single committed route under its own greedy BR
        det = min(greedy_br_attacker(game.route_edges, vuln_fs,
                                     stack_support({r: 1.0}, R), N, k)[1] for r in range(R))
        row["static_det_best_route"] = round(float(det), 4)
        v_fp, d_bar = tabular_fp(game, vuln_fs, k)
        row["tabular_fp_avg"] = round(v_fp, 4)
        row["fp_support_size"] = int((d_bar > 0.01).sum())
        best_stack = min(row[a] for a in arms)
        best_mixed = min(best_stack, row["tabular_fp_avg"])
        row["best_stack_over_det"] = round(best_stack / max(row["static_det_best_route"], 1e-9), 4)
        row["best_mixed_over_det"] = round(best_mixed / max(row["static_det_best_route"], 1e-9), 4)
        row["fp_over_best_stack"] = round(row["tabular_fp_avg"] / max(best_stack, 1e-9), 4)
        rows[f"K{k}"] = row
        print(f"K={k}: {json.dumps(row)}", flush=True)
    OUT["S_saturation"] = rows


def part_x():
    print("\n=== X. exact side on 71-33, K=1..4 ===", flush=True)
    rows = {}
    for k in (1, 2, 3, 4):
        t0 = time.time()
        env = make_multiconvoy_env(("71", "33"), N=N, K=k, k_extra_routes=KX,
                                   edge_vuln_band=BAND, absolute_vuln_norm=True,
                                   menu_select=True, objective="mission")
        game = env.game
        vuln_fs = {frozenset(e): v for e, v in env.edge_vulnerability.items()}
        n_isets = game.payoff.shape[1]
        t_env = time.time() - t0
        R = game.n_routes
        dis = disjoint_subset(game.route_edges)
        q = {r: 1.0 - (1.0 - float(game.payoff[r].max())) ** N for r in range(R)}
        t0 = time.time()
        occs, M = objective_matrix(game, N, "mission")
        t_mat = time.time() - t0
        t0 = time.time()
        v_star, _ = _row_minimiser(M)
        t_lp = time.time() - t0
        arms = {
            "uniform_disjoint": {r: 1.0 for r in dis},
            "inv_vuln_disjoint": {r: 1.0 / max(q[r], 1e-9) for r in dis},
            "uniform_full": {r: 1.0 for r in range(R)},
            "inv_vuln_full": {r: 1.0 / max(q[r], 1e-9) for r in range(R)},
        }
        row = {"n_isets": n_isets, "v_star_exact": round(float(v_star), 4),
               "secs_env": round(t_env, 1), "secs_matrix": round(t_mat, 1),
               "secs_lp": round(t_lp, 1)}
        # exact static_det: best single committed route under the exact best response
        occ_index_det = {tuple(o): i for i, o in enumerate(occs)}
        det_vals = []
        for r in range(R):
            occ = tuple(N if i == r else 0 for i in range(R))
            det_vals.append(float(M[occ_index_det[occ]].max()))
        row["static_det_best_route_exact"] = round(min(det_vals), 4)
        # exact vs greedy stack values (fidelity)
        occ_index = {tuple(o): i for i, o in enumerate(occs)}
        for name, wts in arms.items():
            sup = stack_support(wts, R)
            d_occ = np.zeros(len(occs))
            for occ, p in sup:
                d_occ[occ_index[occ]] = p
            v_exact = float((d_occ @ M).max())
            _, v_greedy = greedy_br_attacker(game.route_edges, vuln_fs, sup, N, k)
            row[name + "_exact"] = round(v_exact, 4)
            row[name + "_greedy"] = round(float(v_greedy), 4)
            row[name + "_fidelity"] = round(abs(v_greedy - v_exact) / max(v_exact, 1e-9), 4)
        rows[f"K{k}"] = row
        print(f"K={k}: {json.dumps(row)}", flush=True)
    OUT["X_exact"] = rows


def part_d():
    print("\n=== D. dynamic K=4 cost model (71-33 kx=8, w=3, tau=0.15) ===", flush=True)
    t0 = time.time()
    env = make_multiconvoy_env(("71", "33"), N=N, K=4, k_extra_routes=KX,
                               edge_vuln_band=BAND, absolute_vuln_norm=True,
                               menu_select=True, objective="mission")
    t_env = time.time() - t0
    t0 = time.time()
    L = stacked_L(env.game, N)
    t_L = time.time() - t0
    R = L.shape[0]
    rng = np.random.default_rng(0)
    # per-sortie softmax_br cost (the trainer calls this once per sortie and per eval step)
    t0 = time.time()
    n_calls = 200
    for _ in range(n_calls):
        counts = np.bincount(rng.integers(0, R, size=3), minlength=R).astype(float)
        softmax_br(counts, L, 0.15)
    t_call = (time.time() - t0) / n_calls
    # eval-style loop (2000 steps in the real eval; net forward excluded, timed separately)
    t0 = time.time()
    window = []
    for _ in range(500):
        counts = np.bincount(window[-3:] if window else [], minlength=R).astype(float)
        br = softmax_br(counts, L, 0.15)
        window.append(int(rng.integers(0, R)))
    t_eval500 = time.time() - t0
    t0 = time.time()
    refs = oracle_refs(L, 0.15, 3)
    t_refs = time.time() - t0
    proj_train = 8000 * t_call
    proj_evals = 16 * (2000 / 500) * t_eval500
    row = {"n_isets": int(L.shape[1]), "secs_env_build": round(t_env, 1),
           "secs_L_build": round(t_L, 1), "secs_per_softmax": round(t_call, 4),
           "secs_eval500_softmax_part": round(t_eval500, 1),
           "secs_oracle_refs": round(t_refs, 1),
           "iid_eq": round(refs["iid_eq"], 4), "static_det": round(refs["static_det"], 4),
           "v_eq_oneshot": round(refs["v_eq"], 4),
           "projected_softmax_secs_8000_sorties": round(proj_train, 1),
           "projected_eval_softmax_secs_16_evals": round(proj_evals, 1)}
    print(json.dumps(row), flush=True)
    print("  (gen40 ext anchors, K=4 kx=8: opt 0.1386, rotation 0.2152, iid_eq 0.3117)",
          flush=True)
    OUT["D_dyn_k4_cost"] = row


def main():
    t0 = time.time()
    env = make_multiconvoy_env(("71", "33"), N=N, K=1, k_extra_routes=KX,
                               edge_vuln_band=BAND, absolute_vuln_norm=True,
                               menu_select=True, objective="mission")
    game = env.game
    vuln_fs = {frozenset(e): v for e, v in env.edge_vulnerability.items()}
    part_s(game, vuln_fs)
    part_x()
    part_d()
    OUT["total_secs"] = round(time.time() - t0, 1)
    json.dump(OUT, open("models/runs/gen43_consolidation_probe.json", "w"), indent=2)
    print(f"\n[written] models/runs/gen43_consolidation_probe.json "
          f"({OUT['total_secs']} s)", flush=True)


if __name__ == "__main__":
    main()
