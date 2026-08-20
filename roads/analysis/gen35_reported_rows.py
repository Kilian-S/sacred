#!/usr/bin/env python3
"""gen35 reported rows (eval-only): computes two reference rows for the K-boundary act. The
tabular window-Q row is average-cost Q-learning over the R^w window states with the same
interaction budget and analytic loss signal SACRED trains on, no network, greedy policy
evaluated exactly every 500 sorties, best value over 3 seeds. The worst-case-committing row
rolls each cell's best checkpoint 2000 sorties against the pattern-of-life enemy and evaluates
the realised route marginal against the one-shot oracle best response.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python analysis/gen35_reported_rows.py
Writes models/runs/gen35_dyn_kboundary/reported_rows.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from analysis.dyn_exact import build_window_mdp, policy_value_exact
from scripts.train_b1lite1 import (
    build_obs, eval_policy, oracle_refs, pick_route, route_feats, softmax_br, stacked_L)
from src.agents.sac import ProtagonistSAC
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(2)
N, BAND, KX, W, TAU = 3, (0.15, 0.95), 8, 3, 0.15
OUT = Path("models/runs/gen35_dyn_kboundary")
PINNED = {2: dict(v_eq=0.2553, mmc=0.0657), 3: dict(v_eq=0.3829, mmc=0.1018)}


def tabular_q(L, cost, n, R, pw, sorties=8000, eps=0.1, lr=0.2, seed=0, eval_every=500):
    """Average-cost Q-learning (RVI-Q) on the window MDP with sampled experience."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((n, R))
    rho = 0.0
    s = 0
    best = np.inf
    for t in range(1, sorties + 1):
        a = int(rng.integers(R)) if rng.random() < eps else int(Q[s].argmin())
        c = cost[s, a]
        s2 = (s % pw) * R + a
        td = c - rho + Q[s2].min() - Q[s, a]
        Q[s, a] += lr * td
        rho += 0.01 * (c + Q[s2].min() - Q[s].min() - rho)
        s = s2
        if t % eval_every == 0:
            pol = np.zeros((n, R))
            pol[np.arange(n), Q.argmin(axis=1)] = 1.0
            v = policy_value_exact(pol, cost, n, R, pw)
            best = min(best, v)
    return best


def worst_case_row(K, env, L, refs_json):
    hist = refs_json["history"]
    bests = {s: min(h[1] for h in hist[s]) for s in hist}
    seed = min(bests, key=bests.get)
    best_k = min(hist[seed], key=lambda h: h[1])[0]
    ck = OUT / f"K{K}_seed{seed}_ckpts" / f"actor_ep{best_k}.pt"
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, gamma=0.95, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    menu_idx = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
    for net in (prot.actor,):
        net.menu_routes = menu_idx
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    prot.actor.load_state_dict(torch.load(ck, weights_only=True))
    from collections import deque
    window = deque(maxlen=W)
    R = env.game.n_routes
    countsv = np.zeros(R)
    for _ in range(2000):
        counts = np.bincount(list(window), minlength=R).astype(float)
        obs = build_obs(env, menu_idx, route_feats(env, counts, W))
        r = pick_route(prot, obs, R)
        countsv[r] += 1
        window.append(r)
    marg = countsv / countsv.sum()
    wc = float((marg @ L).max())
    return dict(seed=int(seed), ckpt=str(ck), marginal=[round(x, 4) for x in marg],
                worst_case_of_marginal=wc, v_eq_oneshot=PINNED[K]["v_eq"],
                premium=wc / PINNED[K]["v_eq"])


def main():
    out = {}
    for K in (2, 3):
        env = make_multiconvoy_env(od=("71", "33"), N=N, K=K, k_extra_routes=KX,
                                   menu_select=True, edge_vuln_band=BAND, interception_loss=10.0)
        L = stacked_L(env.game, N)
        cost, n, R, pw = build_window_mdp(L, TAU, W)
        tq = [tabular_q(L, cost, n, R, pw, seed=s) for s in (0, 1, 2)]
        hist = {}
        for s in (0, 1, 2):
            d = json.load(open(OUT / f"K{K}_seed{s}.json"))
            hist[s] = d["history"]
        wc = worst_case_row(K, env, L, dict(history=hist))
        out[f"K{K}"] = dict(tabular_q_bests=[round(x, 4) for x in tq],
                            tabular_q_pooled=float(np.mean(tq)), worst_case=wc)
        print(f"K={K}: tabular-Q bests {[round(x,4) for x in tq]} pooled {np.mean(tq):.4f} "
              f"(SACRED pooled {0.0934 if K==2 else 0.1406}; rule {0.0929 if K==2 else 0.1539}; "
              f"exact opt {PINNED[K]['mmc']}) | worst-case marginal {wc['worst_case_of_marginal']:.4f} "
              f"= {wc['premium']:.2f}x one-shot v_eq", flush=True)
    with open(OUT / "reported_rows.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT / "reported_rows.json")


if __name__ == "__main__":
    main()
