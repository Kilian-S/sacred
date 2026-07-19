#!/usr/bin/env python3
"""Critic probe 2 (2026-07-19, ORACLE/EVAL-ONLY): re-base the aerial act's ONE surviving
trained claim (zero-shot frontier-MATCHING on the 6 gated held-out layouts, vs-naive
1.01-1.05) against the COMPLETED baseline family that the 2026-07-19 appendix applied to the
flagship cell and the theatre, but never to the held-out cells themselves.

Per gated held-out layout (holdoutD2100-2105, the exact objects the trainer built):
  * eq, ledger best_naive (the act's family: lane/full-menu x uniform/inv-risk, stack+indep);
  * exhaustive best k-route uniform STACKS k=2..5 (oracle-fitted cap on small-support rules);
  * payoff-BLIND safest-L + max-separation stacks (threat map only, the appendix's blind rule);
  * SACRED's zero-shot absolute at the VALIDATION-selected checkpoint, read from the committed
    v3.1/v3.2 run JSONs (models/runs/gen28_fleet2 + gen28_fleet3), per seed.

Question: does "one policy re-derives best-naive-rule performance on sight" survive when
"best naive" means the completed family rather than the act's lane-descended family?
"""
from __future__ import annotations

import itertools
import json

import numpy as np

from scripts.train_aerial_generalist import make_layout_instance

KS = (2, 3, 4, 5)


def stack_matrix(inst):
    """stackM[r] = mission-BR payoff row of 'all N on route r'."""
    env = inst.env
    R = inst.R
    M = env.obj_matrix
    S = np.zeros((R, M.shape[1]))
    for r in range(R):
        v = [0] * R
        v[r] = inst.N
        S[r] = M[env._occ_index[tuple(v)]]
    return S


def best_k_stacks(stackM, ks=KS, chunk=20000):
    out = {}
    R = stackM.shape[0]
    for k in ks:
        bv, bT = np.inf, None
        it = itertools.combinations(range(R), k)
        while True:
            block = list(itertools.islice(it, chunk))
            if not block:
                break
            idx = np.array(block)
            vals = stackM[idx].mean(axis=1).max(axis=1)
            j = int(vals.argmin())
            if vals[j] < bv:
                bv, bT = float(vals[j]), tuple(block[j])
        out[k] = (bv, bT)
    return out


def blind_stacks(inst, stackM):
    """Payoff-blind: pool = L safest routes by worst single-hazard exposure; greedily
    max-separate k of them by lateral signature; uniform stack. Min over (L, k)."""
    env = inst.env
    R = inst.R
    exp = 1.0 - env.S.min(axis=1)
    sig = np.stack([np.array([n[1] for n in env.game.routes[i]], float) for i in range(R)])
    best = np.inf
    for L in (6, 8, 10, 14, R):
        pool = list(np.argsort(exp)[:L])
        d0 = np.linalg.norm(sig[pool][:, None] - sig[pool][None], axis=2)
        i, j = np.unravel_index(np.argmax(d0), d0.shape)
        for k in (3, 4, 5):
            T = [pool[i], pool[j]]
            while len(T) < k:
                cand = [p for p in pool if p not in T]
                dd = [min(np.linalg.norm(sig[c] - sig[t]) for t in T) for c in cand]
                T.append(cand[int(np.argmax(dd))])
            best = min(best, float(stackM[T].mean(axis=0).max()))
    return best


def val_selected_holdouts(run_json, refs):
    """Validation-selected eval entry -> gated holdout absolutes (the v3.1 selection rule)."""
    d = json.load(open(run_json))
    best, out = np.inf, None
    for e in d["history"]:
        val = e[10]
        vr = np.mean([v / refs[k]["eq"] for k, v in val.items()])
        if vr < best:
            best, out = vr, e[5]
    return out


def main():
    insts = {f"holdoutD{2100 + s}": make_layout_instance(f"holdoutD{2100 + s}", 2100 + s, "dbl")
             for s in range(6)}
    rows = {}
    for name, inst in insts.items():
        sm = stack_matrix(inst)
        bk = best_k_stacks(sm)
        blind = blind_stacks(inst, sm)
        best_fit = min(v for v, _ in bk.values())
        rows[name] = dict(eq=inst.eq, ledger_best_naive=inst.best_naive,
                          best_fit_stack=best_fit, blind=blind,
                          per_k={k: v for k, (v, _) in bk.items()})
        print(f"[{name}] eq={inst.eq:.3f} ledger_best_naive={inst.best_naive:.3f} "
              f"({inst.best_naive/inst.eq:.2f}x) | best-k-fit stack={best_fit:.3f} "
              f"({best_fit/inst.eq:.2f}x) | payoff-blind sep-stack={blind:.3f} "
              f"({blind/inst.eq:.2f}x)", flush=True)

    for run in ("models/runs/gen28_fleet2/seed0.json", "models/runs/gen28_fleet2/seed1.json",
                "models/runs/gen28_fleet2/seed2.json", "models/runs/gen28_fleet3/seed0.json",
                "models/runs/gen28_fleet3/seed1.json", "models/runs/gen28_fleet3/seed2.json"):
        refs = json.load(open(run))["refs"]
        ho = val_selected_holdouts(run, refs)
        vs_ledger, vs_fit, vs_blind, beats_fit, beats_blind = [], [], [], 0, 0
        for name, v in ho.items():
            r = rows[name]
            vs_ledger.append(v / r["ledger_best_naive"])
            vs_fit.append(v / r["best_fit_stack"])
            vs_blind.append(v / r["blind"])
            beats_fit += v < r["best_fit_stack"]
            beats_blind += v < r["blind"]
        print(f"{run}: vs-LEDGER-naive {np.mean(vs_ledger):.3f} | "
              f"vs-BEST-FIT-stack {np.mean(vs_fit):.3f} (beats {beats_fit}/6) | "
              f"vs-BLIND-sep {np.mean(vs_blind):.3f} (beats {beats_blind}/6)", flush=True)


if __name__ == "__main__":
    main()
