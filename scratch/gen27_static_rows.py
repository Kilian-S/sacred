#!/usr/bin/env python3
"""gen27 static-baseline rows (ORACLE-EXACT, eval-only; pre-registered by the gen27 ledger
amendment BEFORE results were read).

For each held-out Gdansk instance, the stationary value of a STATIC mixture d against the
pattern-of-life adversary is  V(d) = E_{win ~ d^w} [ d . L . BR(win) ]  (exact enumeration over
R^w windows). Rows: the uniform-disjoint heuristic, the inv-vuln heuristic, the equilibrium
mixture (= iid_eq, the sanity row), and a multi-start projected-gradient LOCAL search for the
static optimum (disclosed as local, an upper bound on how good static play can be).
"""
from __future__ import annotations

import itertools
import json

import numpy as np
import torch

from scripts.train_b1lite1 import softmax_br, stacked_L
from scripts.train_generalist import sample_instances
from src.baselines.multiconvoy_oracle import _row_minimiser

W, TAU = 3, 0.15


def disjoint_subset(route_edges):
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (re_ & used):
            kept.append(i)
            used |= re_
    return kept


def static_value(d, L, tau=TAU, w=W):
    """Exact stationary value of static mixture d vs the softmax-BR window adversary."""
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


def local_static_opt(L, restarts=8, iters=300, lr=0.5, seed=0):
    """Multi-start projected-gradient (softmax-parameterised, finite-diff free: autograd through
    the exact enumeration is overkill; use numerical gradient) local search for the best static
    mixture. Disclosed as LOCAL: a lower bound on iid_eq-family values, not a certificate."""
    R = L.shape[0]
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    for _ in range(restarts):
        z = rng.normal(0, 1, R)
        for _ in range(iters):
            z = z - z.max()
            d = np.exp(z); d /= d.sum()
            v0 = static_value(d, L)
            g = np.zeros(R)
            for r in range(R):
                dz = z.copy(); dz[r] += 1e-3
                dd = np.exp(dz - dz.max()); dd /= dd.sum()
                g[r] = (static_value(dd, L) - v0) / 1e-3
            z = z - lr * g
            if np.linalg.norm(g) < 1e-5:
                break
        d = np.exp(z - z.max()); d /= d.sum()
        v = static_value(d, L)
        if v < best[0]:
            best = (v, d)
    return best[0]


def main():
    torch.set_num_threads(4)
    rows = []
    for it in sample_instances(6, 3, 1, (0.15, 0.95), 8, 0, city="gdansk"):
        L = stacked_L(it.env.game, 3)
        R = L.shape[0]
        dis = disjoint_subset(it.env.game.route_edges)
        uni = np.zeros(R)
        for r in dis:
            uni[r] = 1.0 / len(dis)
        inv = np.zeros(R)
        qs = {r: 1.0 - (1.0 - float(it.env.game.payoff[r].max())) ** 3 for r in dis}
        for r in dis:
            inv[r] = 1.0 / max(qs[r], 1e-9)
        inv /= inv.sum()
        _, eq = _row_minimiser(L)
        row = {"od": f"{it.od[0]}-{it.od[1]}",
               "iid_eq(eq mixture)": round(static_value(eq, L), 4),
               "uniform_disjoint_static": round(static_value(uni, L), 4),
               "inv_vuln_static": round(static_value(inv, L), 4),
               "local_static_opt": round(local_static_opt(L), 4)}
        rows.append(row)
        print(row, flush=True)
    json.dump(rows, open("models/runs/gen27_dyn_generalist/static_rows.json", "w"), indent=2)
    print("[written] models/runs/gen27_dyn_generalist/static_rows.json")


if __name__ == "__main__":
    main()
