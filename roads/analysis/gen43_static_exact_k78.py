#!/usr/bin/env python3
"""gen43 static exact optimum at K = 7 and K = 8 (2026-08-10, ORACLE/EVAL-ONLY, no training).

WHY. The 2026-08-10 extension took the exact static optimum to K = 6 with a dense stacked LP.
K = 7 and K = 8 are past that: the dense LP's own constraint matrix alone would be ~3.1 GB and
~13.9 GB, and the payoff and survival arrays sit beside it, so the direct route peaks around
11 GB and 50 GB. Kilian asked whether they are reachable in theory; they are reachable in
practice, by CONSTRAINT GENERATION.

THE METHOD, and why it is exact. The defender has only R = 11 pure strategies (Prop 3.2: the
mission objective is concave in the occupancy, so stacks suffice), so the LP has 11 unknowns
and, by LP basis size, an optimal attacker mixture supported on at most 12 of the C(43, K)
columns. We therefore never form the full constraint matrix. Instead:

  1. keep a small WORKING SET of columns and solve the exact LP on it, giving (v, d);
  2. SCAN every column of the full game for the one that hurts d most;
  3. if no column exceeds v, then d is optimal against the WHOLE game and v = v* exactly,
     which is a certificate, not an approximation; otherwise add the violating column and
     repeat.

Step 2 is the only expensive part and is done on a cached float32 survival matrix
(R x C(43,K), 1.4 GB at K=7 and 6.4 GB at K=8), built once in chunks so the combination
enumeration never materialises.

ANCHORS. The same solver is run at K = 5 and K = 6 first and must reproduce the dense-LP
values 0.620058 and 0.686494 before any new number is read.

Run: OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
     analysis/gen43_static_exact_k78.py --budgets 5 6 7 8
Artefact: models/runs/gen43_static_exact_k78.json (regenerable, deterministic).
"""
from __future__ import annotations

import argparse
import itertools
import json
import time
from math import comb

import numpy as np

from src.baselines.multiconvoy_oracle import _row_minimiser
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, BAND, KX = 3, (0.15, 0.95), 8
CHUNK = 1_000_000

# dense-LP values from the 2026-08-10 extension, used as anchors
V_STAR_DENSE = {1: 0.127640, 2: 0.255280, 3: 0.382920, 4: 0.510560,
                5: 0.620058, 6: 0.686494}


def build_instance(od=("71", "33")):
    """Route/edge structure only; taken from the cheap K=1 env (the game's candidate edge set
    and route edge sets do not depend on the budget)."""
    env = make_multiconvoy_env(od, N=N, K=1, k_extra_routes=KX, edge_vuln_band=BAND,
                               absolute_vuln_norm=True, menu_select=True, objective="mission")
    game = env.game
    vuln = {frozenset(e): float(v) for e, v in env.edge_vulnerability.items()}
    edges = sorted({e for re_ in game.route_edges for e in re_}, key=lambda fs: sorted(fs))
    eidx = {e: i for i, e in enumerate(edges)}
    R, E = game.n_routes, len(edges)
    # s[r, e] = probability the stacked fleet survives edge e, = (1 - p_e)^N on the route, else 1
    s = np.ones((R, E), dtype=np.float64)
    for r, re_ in enumerate(game.route_edges):
        for e in re_:
            s[r, eidx[e]] = (1.0 - vuln[e]) ** N
    return R, E, s, game


def survival_matrix(s, E, k, verbose=True):
    """Cached float32 (R, C(E,k)) survival matrix, built in chunks."""
    R = s.shape[0]
    n = comb(E, k)
    out = np.empty((R, n), dtype=np.float32)
    it = itertools.combinations(range(E), k)
    pos, t0 = 0, time.time()
    while pos < n:
        m = min(CHUNK, n - pos)
        idx = np.fromiter(itertools.chain.from_iterable(itertools.islice(it, m)),
                          dtype=np.int8, count=m * k).reshape(m, k)
        acc = np.ones((R, m), dtype=np.float64)
        for j in range(k):
            acc *= s[:, idx[:, j]]
        out[:, pos:pos + m] = acc.astype(np.float32)
        pos += m
        if verbose and (pos // CHUNK) % 20 == 0:
            print(f"    ... {pos:,}/{n:,} ({time.time() - t0:.0f} s)", flush=True)
    return out, n, time.time() - t0


def solve_exact(surv, R, tol=1e-12, max_rounds=200, verbose=True):
    """Constraint generation. Returns (v_star, d, n_rounds, working_set_size)."""
    # seed the working set with the columns that hurt a few natural mixtures most
    seeds = [np.ones(R) / R]
    cols = []
    for d0 in seeds:
        vals = 1.0 - (d0.astype(np.float32) @ surv)
        cols.append(int(np.argmax(vals)))
    cols = sorted(set(cols))
    for rnd in range(1, max_rounds + 1):
        L_J = 1.0 - surv[:, cols].astype(np.float64)
        v, d = _row_minimiser(L_J)
        vals = 1.0 - (d.astype(np.float32) @ surv)          # exact scan over EVERY column
        j = int(np.argmax(vals))
        best = float(vals[j])
        if verbose:
            print(f"    round {rnd:>2}: |J|={len(cols):>3}  v={v:.9f}  "
                  f"most-violating={best:.9f}  gap={best - v:.3e}", flush=True)
        if best <= v + 1e-9:
            return float(v), d, rnd, len(cols), float(best - v)
        if j in cols:                                        # numerical stall guard
            return float(v), d, rnd, len(cols), float(best - v)
        cols.append(j)
        cols.sort()
    raise RuntimeError("constraint generation did not certify within max_rounds")


def greedy_column(d, s, k):
    """Exact-in-practice warm-start column: greedily pick the k edges that hurt mixture d most.
    Used only to SEED the working set; optimality is certified by the streaming scan, never
    by this."""
    R, E = s.shape
    acc, chosen = np.ones(R), []
    for _ in range(k):
        best_e, best_v = -1, -np.inf
        for e in range(E):
            if e in chosen:
                continue
            v = 1.0 - float(d @ (acc * s[:, e]))
            if v > best_v:
                best_v, best_e = v, e
        chosen.append(best_e)
        acc = acc * s[:, best_e]
    return acc, chosen


def solve_exact_streaming(s, E, k, verbose=True):
    """Constraint generation without caching the survival matrix: the working set is warm
    started by greedy best responses, and optimality is certified by full chunked scans over
    every one of the C(E,k) columns. Memory is O(R x CHUNK)."""
    R = s.shape[0]
    n = comb(E, k)
    cols = [np.ones(R)]                       # empty interdiction set, a valid column
    for _ in range(60):                       # phase A: cheap greedy warm start
        L_J = 1.0 - np.column_stack(cols)
        v, d = _row_minimiser(L_J)
        acc, _ = greedy_column(d, s, k)
        if any(np.allclose(acc, c) for c in cols):
            break
        cols.append(acc)
    for rnd in range(1, 12):                  # phase B: certified full scans
        L_J = 1.0 - np.column_stack(cols)
        v, d = _row_minimiser(L_J)
        d32 = d.astype(np.float32)
        it = itertools.combinations(range(E), k)
        pos, best_v, best_edges, t0 = 0, -np.inf, None, time.time()
        while pos < n:
            m = min(CHUNK, n - pos)
            idx = np.fromiter(itertools.chain.from_iterable(itertools.islice(it, m)),
                              dtype=np.int8, count=m * k).reshape(m, k)
            acc = np.ones((R, m), dtype=np.float32)
            for j in range(k):
                acc *= s[:, idx[:, j]].astype(np.float32)
            vals = 1.0 - (d32 @ acc)
            jj = int(np.argmax(vals))
            if float(vals[jj]) > best_v:
                best_v, best_edges = float(vals[jj]), idx[jj].astype(int).copy()
            pos += m
        # the scan runs in float32 for speed (absolute precision ~5e-8 at these magnitudes),
        # so the winning column is re-evaluated in float64 before the certificate is applied
        best_acc = np.ones(R)
        for e in best_edges:
            best_acc = best_acc * s[:, int(e)]
        best_exact = 1.0 - float(d @ best_acc)
        if verbose:
            print(f"    scan {rnd}: v={v:.9f}  most-violating={best_exact:.9f}  "
                  f"gap={best_exact - v:.3e}  ({time.time() - t0:.0f} s, {n:,} columns)",
                  flush=True)
        if best_exact <= v + 1e-9:
            return float(v), d, rnd, len(cols), float(best_exact - v)
        cols.append(best_acc)
    raise RuntimeError("streaming constraint generation did not certify")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream", action="store_true",
                    help="certify by chunked scans instead of caching the survival matrix")
    ap.add_argument("--budgets", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--od", nargs=2, default=["71", "33"])
    ap.add_argument("--out", default="models/runs/gen43_static_exact_k78.json")
    args = ap.parse_args()

    R, E, s, game = build_instance(tuple(args.od))
    print(f"instance {args.od}: R={R} routes, |E|={E} candidate edges", flush=True)
    out = {"R": R, "n_edges": E}

    for k in args.budgets:
        n = comb(E, k)
        print(f"\n=== K = {k}: {n:,} interdiction sets, "
              f"survival cache {R * n * 4 / 1e9:.2f} GB ===", flush=True)
        t0 = time.time()
        if args.stream:
            n_isets, t_build = n, 0.0
            v, d, rounds, nj, gap = solve_exact_streaming(s, E, k)
            surv = None
        else:
            surv, n_isets, t_build = survival_matrix(s, E, k)
            print(f"  survival matrix built in {t_build:.0f} s", flush=True)
            v, d, rounds, nj, gap = solve_exact(surv, R)
        t_tot = time.time() - t0
        row = {"n_isets": int(n_isets), "v_star": round(v, 6),
               "certificate_gap": gap, "rounds": rounds, "working_set": nj,
               "secs_build": round(t_build, 1), "secs_total": round(t_tot, 1),
               "cache_gb": round(R * n * 4 / 1e9, 3),
               "defender_mixture": [round(float(x), 6) for x in d],
               "support": int((d > 1e-6).sum())}
        if k in V_STAR_DENSE:
            row["v_star_dense_lp"] = V_STAR_DENSE[k]
            row["anchor_deviation"] = abs(v - V_STAR_DENSE[k])
            ok = "OK" if row["anchor_deviation"] < 1e-5 else "MISMATCH"
            print(f"  ANCHOR vs dense LP {V_STAR_DENSE[k]}: "
                  f"dev {row['anchor_deviation']:.2e}  [{ok}]", flush=True)
        print(f"  v* = {v:.6f}   (certified, gap {gap:.2e}; {rounds} rounds, "
              f"{nj} columns of {n_isets:,}; {t_tot:.0f} s)", flush=True)
        out[f"K{k}"] = row
        del surv
        json.dump(out, open(args.out, "w"), indent=2)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[written] {args.out}", flush=True)


if __name__ == "__main__":
    main()
