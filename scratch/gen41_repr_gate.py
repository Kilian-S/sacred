#!/usr/bin/env python3
"""gen41 ACT 2, GATE 1 (ORACLE-ONLY): the representability certificate at (w=3, K=2) on
the 24 reviewed pools. Pre-registered in gen41_deepwindow_zst.md (ACT 2 section).

Witness values are EXACT: deterministic policies induce deterministic window chains, whose
long-run cost is the best reachable cycle mean (multi-start cycle walk).

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     scratch/gen41_repr_gate.py
Writes models/runs/gen41_repr_gate.json
"""
from __future__ import annotations

import itertools
import json
import time

import numpy as np
import torch

from scratch.critique_followup_probes import disjoint_subset, rotation_value
from scratch.dyn_exact import greedy_policy_from_rvi, karp_mmc
from scratch.gen40_dyn_sensitivity import (
    BAND, N, TAU, best_rotation, enum_windows, inv_vuln_dist, static_stationary,
    uniform_dist, window_losses)
from scratch.gen41_pool_screen import ext_rotation_value, iid_eq_exact
from scripts.train_b1lite1 import stacked_L
from scripts.train_dyn_generalist import load_pool_file
from src.baselines.multiconvoy_oracle import _row_minimiser

torch.set_num_threads(1)
W, K, KX = 3, 2, 12
GRID_COST = (-2.0, -1.0, 0.0, 1.0, 2.0)
GRID_VULN = (-6.0, -4.0, -2.0, -1.0, 0.0)
GRID_FREQ = (-40.0, -25.0, -15.0, -8.0, -4.0, -2.0, 0.0)


def mm(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else np.zeros_like(x)


def build_cells(game, L):
    """Window enumeration (full menu, w=3), per-window per-route losses, and the
    normalised static feature columns the trained head sees."""
    R = L.shape[0]
    dec, counts = enum_windows(list(range(R)), W, R)
    lw = window_losses(counts, L, TAU)
    cost_n = mm(game.travel_cost)
    vuln_n = mm(L.max(axis=1))
    return dec, counts, lw, cost_n, vuln_n, R


def cycle_value(act_of_state, lw, R, starts, want_states=False):
    """Exact long-run cost of a deterministic window policy: best reachable cycle mean.
    Fewer starts can only miss better cycles, so witness values stay conservative."""
    best = np.inf
    pw = R ** (W - 1)
    visited = set()
    for s0 in starts:
        s = int(s0)
        seen = {}
        path = []
        while s not in seen:
            seen[s] = len(path)
            a = int(act_of_state[s])
            path.append(lw[s, a])
            s = (s % pw) * R + a
        if want_states:
            visited.update(seen.keys())
        i = seen[s]
        cyc = path[i:]
        best = min(best, float(np.mean(cyc)))
    return (best, visited) if want_states else best


def linear_witness(counts, lw, cost_n, vuln_n, R, starts):
    """Best deterministic argmax policy over score_r = t1*cost + t2*vuln + t3*freq, on the
    theta grid plus a local refinement pass. Returns (value, theta)."""
    freq = counts / W                                     # [n_states, R]
    best_v, best_t = np.inf, None
    def value_of(t1, t2, t3):
        scores = t1 * cost_n[None, :] + t2 * vuln_n[None, :] + t3 * freq
        acts = scores.argmax(axis=1)
        return cycle_value(acts, lw, R, starts)
    for t1 in GRID_COST:
        for t2 in GRID_VULN:
            for t3 in GRID_FREQ:
                v = value_of(t1, t2, t3)
                if v < best_v:
                    best_v, best_t = v, (t1, t2, t3)
    for _ in range(2):                                    # local refinement
        t1, t2, t3 = best_t
        for d1 in (-0.5, 0.0, 0.5):
            for d2 in (-0.5, 0.0, 0.5):
                for d3 in (-2.0, -1.0, 0.0, 1.0, 2.0):
                    v = value_of(t1 + d1, t2 + d2, t3 + d3)
                    if v < best_v:
                        best_v, best_t = v, (t1 + d1, t2 + d2, t3 + d3)
    return best_v, best_t


def count_witness(dec, counts, lw, R, starts, seeds, max_sweeps=6):
    """Coordinate descent over count-class -> action policies, seeded from projections;
    exact evaluation per candidate; each sweep only touches classes VISITED by the current
    policy's walks (the rest cannot change its value). Returns the best value found."""
    keys = [tuple(c) for c in counts.astype(int).tolist()]
    classes = {}
    for si, k in enumerate(keys):
        classes.setdefault(k, []).append(si)
    best_v = np.inf
    for seed_acts in seeds:
        pol = {}
        for k, members in classes.items():
            acts = [int(seed_acts[s]) for s in members]
            pol[k] = max(set(acts), key=acts.count)
        act_of_state = np.array([pol[k] for k in keys])
        v, visited = cycle_value(act_of_state, lw, R, starts, want_states=True)
        for _ in range(max_sweeps):
            improved = False
            active = {keys[s] for s in visited}
            for k in active:
                cur = pol[k]
                for a in range(R):
                    if a == cur:
                        continue
                    for s in classes[k]:
                        act_of_state[s] = a
                    v2 = cycle_value(act_of_state, lw, R, starts)
                    if v2 < v - 1e-12:
                        v, cur = v2, a
                        improved = True
                    else:
                        for s in classes[k]:
                            act_of_state[s] = cur
                if pol[k] != cur:
                    pol[k] = cur
                    for s in classes[k]:
                        act_of_state[s] = cur
            if not improved:
                break
            v, visited = cycle_value(act_of_state, lw, R, starts, want_states=True)
        best_v = min(best_v, v)
    return best_v


def composed_exact_w3(it_L, route_edges, wp):
    """Exact composed anti-repeat at defender window wp vs the w=3 enemy (corridor chain
    over max(wp, 3)-windows)."""
    L = it_L
    R = L.shape[0]
    dis = disjoint_subset([set(e) for e in route_edges])
    b = len(dis)
    ws = max(wp, W)
    n = b ** ws
    dec = np.empty((n, ws), dtype=np.int64)
    x = np.arange(n)
    for i in range(ws):
        dec[:, ws - 1 - i] = x % b
        x = x // b
    counts3 = np.zeros((n, R))
    for j in range(b):
        counts3[:, dis[j]] = (dec[:, -W:] == j).sum(axis=1)
    lw = window_losses(counts3, L, TAU)[:, dis]
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
    t0 = time.time()
    train, test = load_pool_file("models/runs/gen41_pool.json", N, K, BAND, KX, 0)
    rows = []
    for tag, pool in (("train", train), ("heldout", test)):
        for it in pool:
            t1 = time.time()
            game = it.env.game
            L = stacked_L(game, N)
            dec, counts, lw, cost_n, vuln_n, R = build_cells(game, L)
            n = R ** W
            dis = disjoint_subset([set(e) for e in game.route_edges])
            v_eq, eq = _row_minimiser(L)
            iid = iid_eq_exact(eq, L, w=W)
            opt_full = karp_mmc(lw, n, R, R ** (W - 1))
            core_states = [s for s in range(n)
                           if all(x in dis for x in dec[s])]
            opt_core_cost = lw[:, dis]
            # corridor-restricted optimum on its own 27-state graph
            dec_c, counts_c = enum_windows(dis, W, R)
            lw_c = window_losses(counts_c, L, TAU)
            opt_core = karp_mmc(lw_c[:, dis], len(dis) ** W, len(dis),
                                len(dis) ** (W - 1))
            comp = {wp: composed_exact_w3(L, game.route_edges, wp)
                    for wp in (1, 2, 3, 4)}
            comp_best = min(comp.values())
            rot = best_rotation(dis, L, TAU, W)
            ext, _ = ext_rotation_value(game, L, dis, w=W)
            rng = np.random.default_rng(0)
            frot = min(rotation_value(o, L, TAU, W) for o in
                       [list(range(R))] + [list(rng.permutation(R)) for _ in range(20)])
            starts = list(rng.integers(0, n, 30)) + core_states[:10]
            lin_v, lin_t = linear_witness(counts, lw, cost_n, vuln_n, R, starts)
            opt_pol = greedy_policy_from_rvi(lw, n, R, R ** (W - 1)).argmax(axis=1)
            lin_scores = (lin_t[0] * cost_n[None, :] + lin_t[1] * vuln_n[None, :]
                          + lin_t[2] * counts / W)
            cnt_v = count_witness(dec, counts, lw, R, starts,
                                  seeds=[opt_pol, lin_scores.argmax(axis=1)])
            row = dict(pool=tag, city=it.city, od=list(it.od), R=R,
                       iid_eq=float(iid), opt_full=opt_full, opt_core=opt_core,
                       composed_best=comp_best,
                       composed_by_wp={k: float(v) for k, v in comp.items()},
                       rotation=rot, ext_rotation=ext, full_rotation=frot,
                       linear_witness=lin_v, linear_theta=list(lin_t),
                       count_witness=cnt_v,
                       lin_beats_composed=bool(lin_v < comp_best),
                       lin_over_cap=lin_v / iid, comp_over_cap=comp_best / iid,
                       opt_over_cap=opt_full / iid)
            rows.append(row)
            print(f"{tag} {it.city} {it.od}: cap {iid:.3f} optF {opt_full:.4f} "
                  f"optC {opt_core:.4f} | comp {comp_best:.4f} rot {rot:.4f} "
                  f"ext {ext:.4f} | LIN {lin_v:.4f} (th {lin_t}) CNT {cnt_v:.4f} | "
                  f"{'LIN BEATS COMP' if lin_v < comp_best else 'lin loses'} "
                  f"({round(time.time() - t1, 1)}s)", flush=True)
    ho = [r for r in rows if r["pool"] == "heldout"]
    tr = [r for r in rows if r["pool"] == "train"]
    ho_wins = sum(r["lin_beats_composed"] for r in ho)
    tr_wins = sum(r["lin_beats_composed"] for r in tr)
    pooled_lin = float(np.mean([r["lin_over_cap"] for r in ho]))
    pooled_comp = float(np.mean([r["comp_over_cap"] for r in ho]))
    verdict = "PASS" if (ho_wins >= 4 and tr_wins >= 12 and pooled_lin < pooled_comp) \
        else "FAIL"
    print(f"\nGATE 1: held-out lin-beats-composed {ho_wins}/6; train {tr_wins}/18; "
          f"pooled held-out lin {pooled_lin:.3f} vs composed {pooled_comp:.3f} "
          f"-> {verdict}", flush=True)
    with open("models/runs/gen41_repr_gate.json", "w") as f:
        json.dump(dict(rows=rows, ho_wins=ho_wins, tr_wins=tr_wins,
                       pooled_lin=pooled_lin, pooled_comp=pooled_comp,
                       verdict=verdict, secs=round(time.time() - t0, 1)), f, indent=1)
    print(f"wrote models/runs/gen41_repr_gate.json ({round(time.time() - t0, 1)}s)",
          flush=True)


if __name__ == "__main__":
    main()
