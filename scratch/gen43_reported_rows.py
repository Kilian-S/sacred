#!/usr/bin/env python3
"""gen43 REPORTED rows (eval-only; pre-registered ungated in experiments/gen43_unified_kboundary.md):

1. TABULAR window-Q at matched budget (8000 sorties, same analytic loss signal), greedy
   policy evaluated exactly every 500 sorties, best value, 3 seeds, at the NEW dynamic
   cells K=1 and K=4 (the gen35 machinery verbatim).
2. WORST-CASE COMMITTING row at K=1 and K=4: best seed's best checkpoint rolled 2000
   sorties vs the pattern-of-life enemy; realised route marginal scored against the
   one-shot oracle best response, beside the one-shot v_eq.

Pinned refs (exact): K=1 v_eq 0.1276, optimum 0.0313; K=4 v_eq 0.5106, optimum 0.1386
(gen43 probe + gen40 ext artefacts; the trainer's internal history_opt is defective and
never cited).

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    scratch/gen43_reported_rows.py
Writes models/runs/gen43_unified/reported_rows.json
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from scratch.dyn_exact import build_window_mdp, policy_value_exact
from scratch.gen35_reported_rows import tabular_q
from scripts.train_b1lite1 import build_obs, pick_route, route_feats
from src.agents.sac import ProtagonistSAC
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

torch.set_num_threads(2)
N, BAND, KX, W, TAU = 3, (0.15, 0.95), 8, 3, 0.15
OUT = Path("models/runs/gen43_unified")
PINNED = {1: dict(v_eq=0.1276, mmc=0.0313, rule=0.0387, sacred=0.0462),
          4: dict(v_eq=0.5106, mmc=0.1386, rule=0.2152, sacred=0.1820),
          5: dict(v_eq=0.6201, mmc=0.1756, rule=0.2743, sacred=0.2175),
          6: dict(v_eq=0.6865, mmc=0.2121, rule=0.3295, sacred=0.2638)}
KS = (5, 6)  # cells to (re)compute this invocation; existing artefact keys are preserved


def worst_case_row(K, env, L):
    hist = {}
    for s in (0, 1, 2):
        d = json.load(open(OUT / f"dyn_K{K}_seed{s}.json"))
        hist[s] = d["history"]
    bests = {s: min(h[1] for h in hist[s]) for s in hist}
    seed = min(bests, key=bests.get)
    best_k = min(hist[seed], key=lambda h: h[1])[0]
    ck = OUT / f"dyn_K{K}_seed{seed}_ckpts" / f"actor_ep{best_k}.pt"
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2, heads=4,
                          reward_scale=1.0, lr_actor=3e-4, gamma=0.95, autotune_alpha=True,
                          alpha_init=1.0, device="cpu")
    menu_idx = [torch.tensor(r, dtype=torch.long) for r in env.menu_route_node_idx()]
    for net in (prot.actor,):
        net.menu_routes = menu_idx
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    prot.actor.load_state_dict(torch.load(ck, weights_only=True))
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
                worst_case_of_marginal=round(wc, 4), v_eq_oneshot=PINNED[K]["v_eq"],
                premium=round(wc / PINNED[K]["v_eq"], 3))


def main():
    art = OUT / "reported_rows.json"
    out = json.load(open(art)) if art.exists() else {}
    for K in KS:
        env = make_multiconvoy_env(od=("71", "33"), N=N, K=K, k_extra_routes=KX,
                                   menu_select=True, edge_vuln_band=BAND,
                                   interception_loss=10.0)
        # closed form, verified equal to the trainer's stacked_L to 6.7e-16 (high-K probe)
        L = 1.0 - (1.0 - env.game.payoff) ** N
        cost, n, R, pw = build_window_mdp(L, TAU, W)
        tq = [tabular_q(L, cost, n, R, pw, seed=s) for s in (0, 1, 2)]
        wc = worst_case_row(K, env, L)
        out[f"K{K}"] = dict(tabular_q_bests=[round(x, 4) for x in tq],
                            tabular_q_pooled=round(float(np.mean(tq)), 4), worst_case=wc)
        p = PINNED[K]
        print(f"K={K}: tabular-Q bests {[round(x,4) for x in tq]} pooled {np.mean(tq):.4f} "
              f"(SACRED pooled {p['sacred']}; rule {p['rule']}; exact opt {p['mmc']}) | "
              f"worst-case marginal {wc['worst_case_of_marginal']:.4f} "
              f"= {wc['premium']:.2f}x one-shot v_eq", flush=True)
    with open(OUT / "reported_rows.json", "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", OUT / "reported_rows.json")


if __name__ == "__main__":
    main()
