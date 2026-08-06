#!/usr/bin/env python3
"""gen41 final evaluation (post-training, ORACLE/EVAL-ONLY): the high-precision pass at the
select-on-train checkpoint of each arm, plus the worst-case one-shot row and the STRONG
check against the corridor-restricted optimum. Binding definitions in
experiments/gen41_deepwindow_zst.md (BINDING AT LAUNCH).

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_final_eval.py
Writes models/runs/gen41_deepwindow/final_eval.json
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import torch

from scripts.train_b1lite1 import softmax_br
from scripts.train_dyn_generalist import (
    build_obs, load_pool_file, pick_route, prep_instance)
from src.agents.sac import ProtagonistSAC

torch.set_num_threads(2)
W, TAU, K, KX, N, BAND = 6, 0.15, 2, 12, 3, (0.15, 0.95)
T_FINAL = 20_000
RUN_DIR = Path("models/runs/gen41_deepwindow")
ARMS = [("seed0", False), ("seed1", False), ("seed2", False), ("seed0_nowin", True)]


def make_prot():
    prot = ProtagonistSAC(node_in_dim=14, edge_in_dim=5, hidden_dim=64, num_layers=2,
                          heads=4, reward_scale=1.0, lr_actor=3e-4, gamma=0.95,
                          autotune_alpha=True, alpha_init=1.0, device="cpu")
    for net in (prot.actor, prot.q1, prot.q2, prot.target_q1, prot.target_q2):
        net.route_feat_w = torch.nn.Parameter(torch.zeros(3))
        net.route_feats = None
    return prot


def rollout(prot, it, n, no_window, seed):
    """High-precision seeded rollout: analytic per-sortie expected loss (Rao-Blackwellised);
    also returns the realised marginal route distribution for the worst-case row."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    window = deque(maxlen=W)
    tot = 0.0
    marg = np.zeros(it.nR)
    for _ in range(n):
        counts = np.bincount(list(window), minlength=it.nR).astype(float)
        obs = build_obs(it, counts, W, no_window=no_window)
        r = pick_route(prot, obs, it.nR)
        tot += float(it.L[r] @ softmax_br(counts, it.L, TAU))
        marg[r] += 1
        window.append(r)
    return tot / n, marg / n


def main():
    import sys
    only = sys.argv[1] if len(sys.argv) > 1 else None
    train, test = load_pool_file("models/runs/gen41_pool.json", N, K, BAND, KX, 0)
    for it in test:
        prep_instance(it, TAU, W, fast=True)
    out = {}
    for arm, no_window in ARMS:
        if only and arm != only:
            continue
        js = json.loads((RUN_DIR / f"{arm.replace('_nowin', '')}"
                         f"{'_nowin' if no_window else ''}.json").read_text())
        sel = js["select_on_train"]
        ck = RUN_DIR / f"{arm}_ckpts" / f"actor_ep{sel['sortie']}.pt"
        prot = make_prot()
        prot.actor.load_state_dict(torch.load(str(ck), weights_only=True))
        rows = []
        for i, it in enumerate(test):
            loss, marg = rollout(prot, it, T_FINAL, no_window, seed=1000 + i)
            wc = float((marg @ it.L).max())          # one-shot BR to the marginal mixture
            rows.append(dict(
                od=list(it.od), loss=loss, iid_eq=float(it.refs["iid_eq"]),
                ratio_cap=loss / float(it.refs["iid_eq"]),
                opt_core=float(it.refs["opt_core"]),
                ratio_opt_core=loss / float(it.refs["opt_core"]),
                at_or_below_opt_core=bool(loss <= float(it.refs["opt_core"])),
                worst_case=wc, v_eq=float(it.refs["v_eq"]),
                wc_over_veq=wc / float(it.refs["v_eq"])))
            print(f"{arm} {it.od}: loss {loss:.4f} cap-ratio {rows[-1]['ratio_cap']:.3f} "
                  f"opt-ratio {rows[-1]['ratio_opt_core']:.3f} wc/veq "
                  f"{rows[-1]['wc_over_veq']:.2f}", flush=True)
        pooled = float(np.mean([r["ratio_cap"] for r in rows]))
        out[arm] = dict(selected_sortie=sel["sortie"],
                        selected_test_ratio_lowprec=sel["test_ratio"],
                        pooled_ratio_cap=pooled,
                        beats_cap=sum(1 for r in rows if r["ratio_cap"] < 1.0),
                        at_or_below_opt_core=sum(1 for r in rows
                                                 if r["at_or_below_opt_core"]),
                        mean_wc_over_veq=float(np.mean([r["wc_over_veq"] for r in rows])),
                        per_od=rows,
                        final_iterate_lowprec=js["history"][-1]["test_ratio"],
                        select_on_test_lowprec=js["select_on_test"]["test_ratio"])
        print(f"== {arm}: pooled cap-ratio {pooled:.4f}, beats cap "
              f"{out[arm]['beats_cap']}/6, at-or-below opt_core "
              f"{out[arm]['at_or_below_opt_core']}/6 ==", flush=True)
    suffix = f"_{only}" if only else ""
    with open(RUN_DIR / f"final_eval{suffix}.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote final_eval{suffix}.json", flush=True)


if __name__ == "__main__":
    main()
