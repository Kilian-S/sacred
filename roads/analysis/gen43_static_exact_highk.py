#!/usr/bin/env python3
"""gen43 static exact extension, K = 5 and K = 6 (oracle/eval-only, no training).

The mission objective is concave in the occupancy (thesis Prop 3.2), so restricting the
defender to stacks leaves the game value unchanged: the exact game value needs only the
R x n_isets stacked matrix rather than the 286 x n_isets full-occupancy matrix (about 26x
smaller, ~85 MB at K=5 and ~540 MB at K=6). This computes, via the stacked LP: (A) an anchor
check that the stacked LP reproduces the banked exact v* at K=1..4 and agrees with the
full-occupancy LP (a numerical check of Prop 3.2); (B) the exact v* at K=5 and K=6; (C) the
exact value of every naive stack and of static_det at K=5 and K=6, with greedy-vs-exact
fidelity; (D) the resulting exact best-mixed-over-det ratio.

Run (single process, all thread pools capped):
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    analysis/gen43_static_exact_highk.py --budgets 1 2 3 4 5 6
Artefact: models/runs/gen43_static_exact_highk.json (regenerable, fully deterministic).
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.baselines.multiconvoy_oracle import (
    _row_minimiser, greedy_br_attacker, objective_matrix)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env

N, BAND, KX = 3, (0.15, 0.95), 8

# Banked anchors that must reproduce before any new number is read.
V_STAR_BANKED = {1: 0.1276, 2: 0.2553, 3: 0.3829, 4: 0.5106}
GREEDY_BANKED = {
    2: {"uniform_disjoint": 0.3288, "inv_vuln_disjoint_budgetmax": 0.2978,
        "uniform_full": 0.3812, "inv_vuln_full_budgetmax": 0.3979},
    3: {"uniform_disjoint": 0.4675, "inv_vuln_disjoint_budgetmax": 0.4556,
        "uniform_full": 0.5014, "inv_vuln_full_budgetmax": 0.5040},
    4: {"uniform_disjoint": 0.6017, "inv_vuln_disjoint_budgetmax": 0.5860,
        "uniform_full": 0.5899, "inv_vuln_full_budgetmax": 0.5852},
    5: {"uniform_disjoint": 0.705, "inv_vuln_disjoint_worstedge": 0.638,
        "uniform_full": 0.666, "inv_vuln_full_worstedge": 0.667},
    6: {"uniform_disjoint": 0.800, "inv_vuln_disjoint_worstedge": 0.766,
        "uniform_full": 0.739, "inv_vuln_full_worstedge": 0.730},
}
DET_BANKED = 0.8325          # best committed route, pinned from K=5 up (finding 4)
BEST_MIXED_OVER_DET_GREEDY = {5: 0.746, 6: 0.829}


def disjoint_subset(route_edges):
    kept, used = [], set()
    for i, re_ in enumerate(route_edges):
        if not (re_ & used):
            kept.append(i)
            used |= re_
    return kept


def stack_support(weights_by_route, R):
    """The multiconvoy support format: (occupancy tuple, probability) pairs."""
    tot = sum(weights_by_route.values())
    return [(tuple(N if i == r else 0 for i in range(R)), w / tot)
            for r, w in weights_by_route.items() if w > 0]


def stacked_matrix(game):
    """L[r, j] = mission failure of stacking all N convoys on route r under interdiction set j.

    Vectorised form of scripts/train_b1lite1.stacked_L (that implementation loops in Python
    over R x n_isets entries, which is fine at K <= 4 and far too slow at 6.1M columns).
    Identity with the loop version is asserted on a column slice below.
    """
    log_surv = np.log(np.clip(1.0 - game.payoff, 1e-300, 1.0))   # [R, n_isets]
    return 1.0 - np.exp(N * log_surv)


def arms_for(game, vuln_fs):
    """The naive stack family, with BOTH inverse-vulnerability conventions.

    DISCLOSURE (found 2026-08-10 by this probe). The banked ladder uses two different
    definitions of the inverse-vulnerability weights, because the consolidation probe's two
    halves computed them differently:

      * `worstedge` (part_s, the K >= 5 rows): weights from each route's WORST SINGLE EDGE,
        1/(1 - (1 - max_e p_e)^N), a property of the map alone and therefore FIXED as K
        varies. This is the thesis's stated definition (Prop floor, appendix E: p_i^* =
        max_{e in E(r_i)} p_e).
      * `budgetmax` (part_x, the K <= 4 rows): weights from max_j payoff[r, j], i.e. the
        worst K-EDGE attack aimed at that route, which coincides with `worstedge` at K = 1
        and diverges above it, becoming near-uniform as K grows.

    Both are computed here at every budget so the divergence is measured rather than assumed.
    """
    R = game.n_routes
    dis = disjoint_subset(game.route_edges)
    # thesis definition: worst single edge on the route, K-independent
    q_edge = {r: 1.0 - (1.0 - max(float(vuln_fs[e]) for e in game.route_edges[r])) ** N
              for r in range(R)}
    # the K-dependent variant the K <= 4 anchors were computed with
    q_budget = {r: 1.0 - (1.0 - float(game.payoff[r].max())) ** N for r in range(R)}
    return {
        "uniform_disjoint": {r: 1.0 for r in dis},
        "inv_vuln_disjoint_worstedge": {r: 1.0 / max(q_edge[r], 1e-9) for r in dis},
        "inv_vuln_disjoint_budgetmax": {r: 1.0 / max(q_budget[r], 1e-9) for r in dis},
        "uniform_full": {r: 1.0 for r in range(R)},
        "inv_vuln_full_worstedge": {r: 1.0 / max(q_edge[r], 1e-9) for r in range(R)},
        "inv_vuln_full_budgetmax": {r: 1.0 / max(q_budget[r], 1e-9) for r in range(R)},
    }


def run_budget(k, do_occupancy_check, od=("71", "33")):
    print(f"\n=== K = {k} " + "=" * 52, flush=True)
    t0 = time.time()
    env = make_multiconvoy_env(od, N=N, K=k, k_extra_routes=KX,
                               edge_vuln_band=BAND, absolute_vuln_norm=True,
                               menu_select=True, objective="mission")
    game = env.game
    vuln_fs = {frozenset(e): v for e, v in env.edge_vulnerability.items()}
    R, n_isets = game.n_routes, game.payoff.shape[1]
    t_env = time.time() - t0
    print(f"  env: R={R}, n_isets={n_isets:,}, {t_env:.1f} s", flush=True)

    t0 = time.time()
    L = stacked_matrix(game)
    t_L = time.time() - t0
    print(f"  stacked matrix: {L.shape}, {L.nbytes / 1e9:.3f} GB, {t_L:.1f} s", flush=True)

    # identity of the vectorised stacked matrix against the committed loop implementation,
    # checked on a column slice (the loop is too slow to run in full at high K)
    from scripts.train_b1lite1 import stacked_L as stacked_L_loop

    class _Slice:
        pass
    sl = _Slice()
    sl.n_routes = R
    sl.payoff = game.payoff[:, :200]
    max_dev = float(np.abs(stacked_L_loop(sl, N) - L[:, :200]).max())
    print(f"  stacked-matrix identity vs committed loop (200 columns): "
          f"max dev {max_dev:.3e}", flush=True)

    t0 = time.time()
    v_star_stacked, d_star = _row_minimiser(L)
    t_lp = time.time() - t0
    print(f"  stacked LP: v* = {v_star_stacked:.6f}  ({t_lp:.1f} s, "
          f"support {(d_star > 1e-6).sum()} of {R})", flush=True)

    row = {"n_isets": int(n_isets), "secs_env": round(t_env, 1),
           "secs_stacked_matrix": round(t_L, 1), "secs_lp": round(t_lp, 1),
           "stacked_matrix_gb": round(L.nbytes / 1e9, 3),
           "stacked_loop_identity_maxdev": max_dev,
           "v_star_stacked": round(float(v_star_stacked), 6),
           "v_star_support": int((d_star > 1e-6).sum()),
           "defender_mixture": [round(float(x), 6) for x in d_star]}

    # A. anchor: the full-occupancy LP must agree (numerical check of Prop 3.2)
    if do_occupancy_check:
        t0 = time.time()
        occs, M = objective_matrix(game, N, "mission")
        v_star_occ, _ = _row_minimiser(M)
        row["v_star_full_occupancy"] = round(float(v_star_occ), 6)
        row["prop_stacks_deviation"] = abs(float(v_star_occ) - float(v_star_stacked))
        row["n_occupancies"] = len(occs)
        row["secs_occupancy_path"] = round(time.time() - t0, 1)
        print(f"  full-occupancy LP: v* = {v_star_occ:.6f} over {len(occs)} occupancies; "
              f"deviation from stacked {row['prop_stacks_deviation']:.3e}", flush=True)

    if k in V_STAR_BANKED:
        dev = abs(float(v_star_stacked) - V_STAR_BANKED[k])
        row["v_star_banked"] = V_STAR_BANKED[k]
        row["v_star_banked_deviation"] = round(dev, 6)
        ok = "OK" if dev < 5e-5 else "MISMATCH"
        print(f"  anchor vs banked v* {V_STAR_BANKED[k]}: dev {dev:.2e}  [{ok}]", flush=True)

    # C. exact values of the naive stacks and of static_det, and greedy fidelity
    arms = arms_for(game, vuln_fs)
    for name, wts in arms.items():
        d = np.zeros(R)
        tot = sum(wts.values())
        for r, w in wts.items():
            d[r] = w / tot
        v_exact = float((d @ L).max())
        _, v_greedy = greedy_br_attacker(game.route_edges, vuln_fs,
                                         stack_support(wts, R), N, k)
        row[name + "_exact"] = round(v_exact, 6)
        row[name + "_greedy"] = round(float(v_greedy), 6)
        row[name + "_fidelity"] = round(abs(float(v_greedy) - v_exact) / max(v_exact, 1e-9), 6)
        banked = GREEDY_BANKED.get(k, {}).get(name)
        extra = ""
        if banked is not None:
            extra = f"  (banked greedy {banked}, dev {abs(float(v_greedy) - banked):.4f})"
        print(f"  {name:<20s} exact {v_exact:.4f}   greedy {float(v_greedy):.4f}   "
              f"fidelity {row[name + '_fidelity'] * 100:.2f}%{extra}", flush=True)

    det_exact = float(L.max(axis=1).min())
    row["static_det_exact"] = round(det_exact, 6)
    for tag in ("worstedge", "budgetmax"):
        members = [a for a in arms if not a.startswith("inv_vuln")
                   or a.endswith(tag)]
        row[f"best_stack_exact_{tag}"] = round(min(row[a + "_exact"] for a in members), 6)
        row[f"best_stack_arm_{tag}"] = min(members, key=lambda a: row[a + "_exact"])
    row["best_stack_exact"] = round(min(row[a + "_exact"] for a in arms), 6)
    row["best_mixed_over_det_exact"] = round(float(v_star_stacked) / max(det_exact, 1e-9), 6)
    row["best_stack_over_det_exact"] = round(row["best_stack_exact"] / max(det_exact, 1e-9), 6)
    print(f"  static_det exact {det_exact:.4f} (banked greedy {DET_BANKED})", flush=True)
    print(f"  best-mixed-over-det EXACT {row['best_mixed_over_det_exact']:.4f}"
          + (f"  (greedy-yardstick banked {BEST_MIXED_OVER_DET_GREEDY[k]})"
             if k in BEST_MIXED_OVER_DET_GREEDY else ""), flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--occupancy-check-max-k", type=int, default=4,
                    help="run the full-occupancy LP as the Prop 3.2 check up to this budget")
    ap.add_argument("--od", nargs=2, default=["71", "33"])
    ap.add_argument("--out", default="models/runs/gen43_static_exact_highk.json")
    args = ap.parse_args()

    out, t0 = {}, time.time()
    for k in args.budgets:
        out[f"K{k}"] = run_budget(k, do_occupancy_check=(k <= args.occupancy_check_max_k), od=tuple(args.od))
        json.dump(out, open(args.out, "w"), indent=2)      # persist per budget
    out["total_secs"] = round(time.time() - t0, 1)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"\n[written] {args.out}  ({out['total_secs']} s)", flush=True)


if __name__ == "__main__":
    main()
