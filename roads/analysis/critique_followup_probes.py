#!/usr/bin/env python3
"""Analytic baselines for the multi-convoy interdiction game (ORACLE/EVAL-ONLY, no training).

Three probes: (A) naive dynamic baselines, deterministic rotation and stochastic anti-repeat
over the edge-disjoint routes, against the pattern-of-life adversary (softmax best-response to
the trailing w-sortie window), compared against the iid-equilibrium and history-optimal values;
(B) full-menu naive stacks (uniform and inverse-vulnerability weighted) at high interdiction
budgets K under the greedy best-response oracle; (C) tabular smooth fictitious play
(multiplicative-weights defender vs greedy best-response attacker, no neural network) using
the same oracle.

Run: PYTHONPATH=. .venv/bin/python analysis/critique_followup_probes.py (single-threaded)
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from scripts.train_b1lite1 import softmax_br, stacked_L
from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import (
    _row_minimiser, greedy_br_attacker, objective_matrix)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(1)
N, BAND, KX, W, TAU = 3, (0.15, 0.95), 8, 3, 0.15
OUT = {}


def disjoint_subset(route_edges):
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (re_ & used):
            kept.append(i)
            used |= re_
    return kept


def static_value(d, L, tau=TAU, w=W):
    R = L.shape[0]
    dL = d @ L
    v = 0.0
    for win in itertools.product(range(R), repeat=w):
        pw = 1.0
        for i in win:
            pw *= d[i]
        if pw > 0:
            v += pw * float(dL @ softmax_br(np.bincount(win, minlength=R).astype(float), L, tau))
    return v


def rotation_value(seq, L, tau=TAU, w=W):
    """Exact stationary per-sortie loss of a deterministic cyclic rotation."""
    m = len(seq)
    R = L.shape[0]
    v = 0.0
    for t in range(m):
        win = [seq[(t - 1 - i) % m] for i in range(w)]
        counts = np.bincount(win, minlength=R).astype(float)
        v += float(L[seq[t]] @ softmax_br(counts, L, tau))
    return v / m


def antirepeat_value(dis, L, tau=TAU, w=W, iters=600):
    """Exact stationary per-sortie loss of 'uniform over disjoint routes not in the last-w
    window' (stochastic anti-repeat), via power iteration on the window Markov chain."""
    R = L.shape[0]
    states = list(itertools.product(dis, repeat=w))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    P = np.zeros((n, n))
    c = np.zeros(n)
    for s in states:
        i = idx[s]
        allowed = [r for r in dis if r not in s] or list(dis)
        counts = np.bincount(s, minlength=R).astype(float)
        br = softmax_br(counts, L, tau)
        c[i] = float(np.mean([L[r] @ br for r in allowed]))
        for r in allowed:
            P[i, idx[s[1:] + (r,)]] += 1.0 / len(allowed)
    pi = np.ones(n) / n
    for _ in range(iters):
        pi = pi @ P
    return float(pi @ c)


def history_opt_rvi(L, tau=TAU, w=W, iters=4000, tol=1e-10):
    """Exact average-cost optimum of the window MDP (RVI over R^w states) on this L; the
    same-convention anchor for the rotation/anti-repeat rows."""
    R = L.shape[0]
    states = list(itertools.product(range(R), repeat=w))
    idx = {s: i for i, s in enumerate(states)}
    n = len(states)
    cost = np.zeros((n, R))
    nxt = np.zeros((n, R), dtype=int)
    for s in states:
        i = idx[s]
        br = softmax_br(np.bincount(s, minlength=R).astype(float), L, tau)
        for a in range(R):
            cost[i, a] = float(L[a] @ br)
            nxt[i, a] = idx[s[1:] + (a,)]
    h = np.zeros(n)
    g = 0.0
    for _ in range(iters):
        q = cost + h[nxt]
        h_new = q.min(axis=1)
        g_new = h_new[0]
        h_next = h_new - g_new
        if np.max(np.abs(h_next - h)) < tol:
            h = h_next
            g = g_new
            break
        h, g = h_next, g_new
    return float(g)


def part_a():
    print("=== A. naive-DYNAMIC baselines vs the pattern-of-life adversary (w=3, tau=0.15) ===")
    rows = []
    env = make_multiconvoy_env(("35", "159"), N=N, K=1, k_extra_routes=KX, edge_vuln_band=BAND,
                               absolute_vuln_norm=True, menu_select=True, objective="mission")
    insts = [("35-159 (gen19 instance)", env.game)]
    for it in sample_instances(6, N, 1, BAND, KX, 0, city="gdansk"):
        insts.append((f"gdansk {it.od[0]}-{it.od[1]} (gen27 held-out)", it.env.game))
    rng = np.random.default_rng(0)
    for name, game in insts:
        L = stacked_L(game, N)
        dis = disjoint_subset(game.route_edges)
        _, eq = _row_minimiser(L)
        iid_eq = static_value(eq, L)
        rot_plain = rotation_value(dis, L)
        perms = [list(rng.permutation(dis)) for _ in range(20)]
        rot_best = min(rotation_value(p, L) for p in perms)
        anti = antirepeat_value(dis, L)
        hopt = history_opt_rvi(L)
        row = {"instance": name, "m": len(dis), "iid_eq": round(iid_eq, 4),
               "history_opt(this L)": round(hopt, 4),
               "rotation_plain": round(rot_plain, 4),
               "rotation_best_of_20_orders": round(rot_best, 4),
               "antirepeat_uniform": round(anti, 4),
               "rot/iid_eq": round(rot_plain / iid_eq, 3),
               "anti/iid_eq": round(anti / iid_eq, 3),
               "anti/history_opt": round(anti / max(hopt, 1e-9), 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    OUT["A_dynamic_naive"] = rows


def stack_support(weights_by_route, R):
    tot = sum(weights_by_route.values())
    return [(tuple(N if i == r else 0 for i in range(R)), w / tot)
            for r, w in weights_by_route.items() if w > 0]


def part_b():
    print("\n=== B. FULL-MENU naive stacks at K=5/6 on 71-33 (greedy yardstick) ===")
    env = make_multiconvoy_env(("71", "33"), N=N, K=1, k_extra_routes=KX, edge_vuln_band=BAND,
                               absolute_vuln_norm=True, menu_select=True, objective="mission")
    game = env.game
    vuln_fs = {frozenset(k): v for k, v in env.edge_vulnerability.items()}
    R = game.n_routes
    dis = disjoint_subset(game.route_edges)
    q = {r: 1.0 - (1.0 - float(game.payoff[r].max())) ** N for r in range(R)}
    arms = {
        "uniform_disjoint (ledger sanity)": {r: 1.0 for r in dis},
        "inv_vuln_disjoint (ledger sanity)": {r: 1.0 / max(q[r], 1e-9) for r in dis},
        "uniform_FULL_menu": {r: 1.0 for r in range(R)},
        "inv_vuln_FULL_menu": {r: 1.0 / max(q[r], 1e-9) for r in range(R)},
    }
    rows = {}
    for k in (5, 6):
        rows[f"K{k}"] = {}
        for name, wts in arms.items():
            _, v = greedy_br_attacker(game.route_edges, vuln_fs, stack_support(wts, R), N, k)
            rows[f"K{k}"][name] = round(float(v), 3)
        print(f"K={k}: {json.dumps(rows[f'K{k}'])}", flush=True)
    print("  (SACRED anchors from gen26: K=5 best-ckpt 0.667 +/- 0.016; K=6 0.718 single seed)")
    OUT["B_full_menu_highK"] = rows
    return game, vuln_fs, dis, q


def loss_vector(game, vuln_fs, S):
    """Per-stacked-route mission failure under interdiction set S (list of frozensets)."""
    R = game.n_routes
    out = np.zeros(R)
    for r in range(R):
        surv = 1.0
        for e in S:
            if e in game.route_edges[r]:
                surv *= 1.0 - vuln_fs[e]
        q_r = 1.0 - surv
        out[r] = 1.0 - (1.0 - q_r) ** N
    return out


def part_c(game, vuln_fs, dis, q):
    print("\n=== C. TABULAR smooth FP with the same greedy-BR oracle (no net) on 71-33 ===")
    R = game.n_routes
    rows = {}
    for k in (5, 6):
        for init_name, init in (("uniform_full", np.ones(R) / R),):
            logw = np.log(init + 1e-12)
            d_bar = init.copy()
            evals = []
            T, eta = 300, 0.5
            for t in range(1, T + 1):
                S, _ = greedy_br_attacker(game.route_edges, vuln_fs,
                                          stack_support({r: d_bar[r] for r in range(R)}, R), N, k)
                l = loss_vector(game, vuln_fs, S)
                logw -= eta * l
                logw -= logw.max()
                d = np.exp(logw)
                d /= d.sum()
                d_bar = (d_bar * t + d) / (t + 1)
                if t % 25 == 0:
                    _, v = greedy_br_attacker(game.route_edges, vuln_fs,
                                              stack_support({r: d_bar[r] for r in range(R)},
                                                            R), N, k)
                    evals.append(round(float(v), 4))
            _, v_final = greedy_br_attacker(game.route_edges, vuln_fs,
                                            stack_support({r: d_bar[r] for r in range(R)}, R),
                                            N, k)
            rows[f"K{k}"] = {"final_avg_strategy_value": round(float(v_final), 4),
                             "best_eval_along_the_way": min(evals),
                             "eval_trace_every25": evals}
            print(f"K={k} ({init_name}): final {v_final:.4f}; trace {evals}", flush=True)
    print("  (SACRED anchors: K=5 0.667 +/- 0.016; K=6 0.718. Heuristics: see part B.)")
    OUT["C_tabular_fp_greedy"] = rows


def main():
    part_a()
    game, vuln_fs, dis, q = part_b()
    part_c(game, vuln_fs, dis, q)
    json.dump(OUT, open("models/runs/critique_followup_probes.json", "w"), indent=2)
    print("\n[written] models/runs/critique_followup_probes.json")


if __name__ == "__main__":
    main()
