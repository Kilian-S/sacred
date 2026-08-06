#!/usr/bin/env python3
"""gen41 Tier-2 adaptive rows (ORACLE/EVAL-ONLY): the earn-your-knowledge rule family on the
6 held-out Gdansk instances, per the binding definitions in gen41_deepwindow_zst.md.

(i)/(ii) EXP3 (full menu / corridors), loss-form exponential weights,
eta = sqrt(ln n / (n T)), T = 12,000 in place, scored on the final-2,000 tail (generous)
with full-horizon means reported; 5 seeded reps.
(iii) Avoid-where-ambushed: sampled committed set, Bernoulli mission outcome on the stacked
loss, avoid routes hit in the last h in {3,6,12}, best h taken (generous); 5 reps.
(iv) Self-tuned composed: EXACT stationary value at defender window w' in {2,4,6,8}
against the w=6 enemy (chain over max(w',6)-window corridor states), best w' taken.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_tier2_rows.py
Writes models/runs/gen41_deepwindow/tier2_rows.json
"""
from __future__ import annotations

import json
from collections import deque

import numpy as np
import torch

from scripts.train_b1lite1 import softmax_br
from scripts.train_dyn_generalist import load_pool_file, prep_instance
from scratch.critique_followup_probes import disjoint_subset

torch.set_num_threads(1)
W, TAU, K, KX, N, BAND = 6, 0.15, 2, 12, 3, (0.15, 0.95)
T, TAIL, REPS = 12_000, 2_000, 5


class QCache:
    """Memoised adversary response by window-count signature."""

    def __init__(self, L):
        self.L = L
        self.c = {}

    def q(self, counts):
        key = tuple(int(x) for x in counts)
        if key not in self.c:
            self.c[key] = softmax_br(np.asarray(counts, float), self.L, TAU)
        return self.c[key]


def exp3(it, arms, seed):
    rng = np.random.default_rng(seed)
    L = it.L
    n = len(arms)
    eta = np.sqrt(np.log(n) / (n * T))
    cumloss = np.zeros(n)
    window = deque(maxlen=W)
    qc = it.qcache
    losses = np.empty(T)
    for t in range(T):
        z = -eta * cumloss
        z -= z.max()
        p = np.exp(z)
        p /= p.sum()
        a = rng.choice(n, p=p)
        r = arms[a]
        counts = np.bincount(list(window), minlength=it.nR).astype(float)
        q = qc.q(counts)
        loss = float(L[r] @ q)
        losses[t] = loss
        cumloss[a] += loss / p[a]
        window.append(r)
    return float(losses[-TAIL:].mean()), float(losses.mean())


def avoid_ambushed(it, h, seed):
    rng = np.random.default_rng(seed)
    L = it.L
    R = it.nR
    qc = it.qcache
    window = deque(maxlen=W)
    last_hit = np.full(R, -10**9)
    losses = np.empty(T)
    for t in range(T):
        counts = np.bincount(list(window), minlength=it.nR).astype(float)
        q = qc.q(counts)
        ok = [r for r in range(R) if t - last_hit[r] > h]
        if not ok:
            ok = list(range(R))
        r = ok[rng.integers(len(ok))]
        losses[t] = float(L[r] @ q)
        j = rng.choice(len(q), p=q)
        if rng.random() < L[r, j]:
            last_hit[r] = t
        window.append(r)
    return float(losses[-TAIL:].mean()), float(losses.mean())


def composed_exact(it, wp):
    """Exact stationary loss of composed anti-repeat at defender window wp against the w=6
    enemy: damped power iteration over the max(wp, 6)-window corridor chain."""
    L = it.L
    R = it.nR
    dis = disjoint_subset([set(e) for e in it.env.game.route_edges])
    b = len(dis)
    ws = max(wp, W)
    n = b ** ws
    dec = np.empty((n, ws), dtype=np.int64)
    x = np.arange(n)
    for i in range(ws):
        dec[:, ws - 1 - i] = x % b
        x = x // b
    counts6 = np.zeros((n, R))
    for j in range(b):
        counts6[:, dis[j]] = (dec[:, -W:] == j).sum(axis=1)
    qc = it.qcache
    lw = np.empty((n, b))
    for s in range(n):
        q = qc.q(counts6[s])
        lw[s] = (L @ q)[dis]
    inwin = np.zeros((n, b), dtype=bool)
    for j in range(b):
        inwin[:, j] = (dec[:, -wp:] == j).any(axis=1)
    allowed = ~inwin
    allowed[~allowed.any(axis=1)] = True
    na = allowed.sum(axis=1)
    c = np.where(allowed, lw, 0.0).sum(axis=1) / na
    w_a = allowed / na[:, None]
    heads = (np.arange(n) % (b ** (ws - 1))) * b
    pi = np.ones(n) / n
    for _ in range(40_000):
        nxt = np.zeros(n)
        for a in range(b):
            nxt += np.bincount(heads + a, weights=pi * w_a[:, a], minlength=n)
        nxt = 0.5 * nxt + 0.5 * pi
        if np.max(np.abs(nxt - pi)) < 1e-12:
            pi = nxt
            break
        pi = nxt
    return float(pi @ c)


def main():
    _, test = load_pool_file("models/runs/gen41_pool.json", N, K, BAND, KX, 0)
    out = []
    for i, it in enumerate(test):
        prep_instance(it, TAU, W, fast=True)
        it.qcache = QCache(it.L)
        dis = disjoint_subset([set(e) for e in it.env.game.route_edges])
        cap = float(it.refs["iid_eq"])
        row = dict(od=list(it.od), iid_eq=cap, opt_core=float(it.refs["opt_core"]))
        e3m = [exp3(it, list(range(it.nR)), 100 * i + s) for s in range(REPS)]
        e3c = [exp3(it, dis, 200 * i + s) for s in range(REPS)]
        row["exp3_menu_tail"] = float(np.mean([x[0] for x in e3m]))
        row["exp3_menu_full"] = float(np.mean([x[1] for x in e3m]))
        row["exp3_core_tail"] = float(np.mean([x[0] for x in e3c]))
        row["exp3_core_full"] = float(np.mean([x[1] for x in e3c]))
        aw = {h: [avoid_ambushed(it, h, 300 * i + 10 * h + s) for s in range(REPS)]
              for h in (3, 6, 12)}
        row["avoid_best_h_tail"] = float(min(np.mean([x[0] for x in v])
                                             for v in aw.values()))
        row["avoid_by_h_tail"] = {h: float(np.mean([x[0] for x in v]))
                                  for h, v in aw.items()}
        comp = {wp: composed_exact(it, wp) for wp in (2, 4, 6, 8)}
        row["composed_selftuned"] = float(min(comp.values()))
        row["composed_by_wp"] = {k: float(v) for k, v in comp.items()}
        for k in ("exp3_menu_tail", "exp3_core_tail", "avoid_best_h_tail",
                  "composed_selftuned"):
            row[k + "_ratio"] = row[k] / cap
        out.append(row)
        print(f"{it.od}: cap {cap:.3f} | EXP3 menu {row['exp3_menu_tail']:.4f} "
              f"core {row['exp3_core_tail']:.4f} | avoid {row['avoid_best_h_tail']:.4f} "
              f"| composed-tuned {row['composed_selftuned']:.4f} "
              f"(w'={min(comp, key=comp.get)})", flush=True)
    pooled = {k: float(np.mean([r[k + "_ratio"] for r in out]))
              for k in ("exp3_menu_tail", "exp3_core_tail", "avoid_best_h_tail",
                        "composed_selftuned")}
    print("pooled ratios-to-cap:", {k: round(v, 3) for k, v in pooled.items()}, flush=True)
    with open("models/runs/gen41_deepwindow/tier2_rows.json", "w") as f:
        json.dump(dict(rows=out, pooled=pooled), f, indent=1)
    print("wrote tier2_rows.json", flush=True)


if __name__ == "__main__":
    main()
