#!/usr/bin/env python3
"""Critic baseline-completeness probe (2026-07-18, committed 2026-07-19; ORACLE-ONLY).

The rows that RETIRE the gen28 fleet Tier-1 superiority wording and complete the vector
theatre's naive family (ledger appendix of 2026-07-19; the gen26-K=6 lesson applied here).

Part 1 - the v3.x fleet flagship cell (dblpinch_banded_K1_r1.2, N=3 mission, eq ~0.538):
  * exhaustive best k-route uniform STACKS (k=2..6) over the menu: the cap on every
    "pick a few routes and randomise" rule (oracle-fitted, disclosed);
  * payoff-BLIND "safest-L + max-separation-k" stacks (threat map only, no payoff access).
  Expected: best-5 stack ~0.60 (1.12x eq) and the act's own tabular-FP row 0.555 both far
  below SACRED's pooled best-ckpts 0.734-0.746; blind rules ~0.76 (SACRED survives blind
  rules; loses to the oracle-fitted subset class and to its own pre-registered FP row).
  On the lane-less double pinch, all_lane_sets collapses to the weakest family member
  (full-menu uniform), which is how best_naive inflated to 0.754.

Part 2 - the vector theatre headline (K=1, standoff 4 km, N=3 mission, eq ~0.373):
  the complete naive family (full-menu / cover-route / top-k-safest stacks), exhaustive
  best k-route stacks, and payoff-blind greedy-max-separation rules. Expected: the
  lanes-only rows (0.609 = 1.63x) are in fact the best FIXED rule, but the best 3-route
  stack is 0.438 (1.17x) and the blind greedy-separation rule 0.470 (1.26x): the quotable
  theatre gap is ~1.26x (blind) / ~1.07x (oracle-fit), not 1.63-1.78x. No theatre bar may
  be pinned before these rows join the gated baseline.
"""
from __future__ import annotations

import itertools

import numpy as np

from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy

N = 3


def occ_maps(R):
    occs = list(itertools.combinations_with_replacement(range(R), N))
    oidx = {}
    for i, o in enumerate(occs):
        v = [0] * R
        for r in o:
            v[r] += 1
        oidx[tuple(v)] = i
    return occs, oidx


def stack_rows(M, R, oidx):
    """stackM[r] = the mission-BR payoff row of 'all N on route r'."""
    S = np.zeros((R, M.shape[1]))
    for r in range(R):
        v = [0] * R
        v[r] = N
        S[r] = M[oidx[tuple(v)]]
    return S


def stack_val(stackM, d):
    return float((np.asarray(d) @ stackM).max())


def best_k_stacks(stackM, ks=(2, 3, 4, 5, 6)):
    out = {}
    R = stackM.shape[0]
    for k in ks:
        bv, bT = np.inf, None
        for T in itertools.combinations(range(R), k):
            v = stackM[list(T)].mean(axis=0).max()
            if v < bv:
                bv, bT = float(v), T
        out[k] = (bv, bT)
    return out


def part1_fleet_cell():
    from src.envs.aerial_curves import build_curve_menu, build_curved_game, dense_hazard_grid
    from src.envs.aerial_sector import SectorLattice, banded_pmax
    DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
        {(4, j) for j in range(9) if j < 5} | {(8, j) for j in range(9) if j > 3}))
    menu, _ = build_curve_menu(DBL, 1.2, R=40, seed=0)
    centres = dense_hazard_grid(DBL, step=0.5)
    game, S = build_curved_game(DBL, menu, centres, 1, r=1.2, p_max=banded_pmax(centres, DBL.ny))
    sol = solve_multiconvoy(game, N, "mission")
    _, M = objective_matrix(game, N, "mission", 1)
    R = game.n_routes
    _, oidx = occ_maps(R)
    stackM = stack_rows(M, R, oidx)
    print(f"[fleet dblpinch_banded_K1_r1.2] R={R} H={len(centres)} eq={sol.loss_mixed:.4f} "
          f"det={sol.loss_det:.4f} | ledger best_naive 0.754 | SACRED pooled 0.746/0.734/0.742 "
          f"| ledger tabular-FP 0.555")
    for k, (v, T) in best_k_stacks(stackM).items():
        print(f"  best {k}-route uniform stack: {v:.4f} ({v/sol.loss_mixed:.2f}x eq)")
    # payoff-blind rules: threat map only (exposure = worst single-hazard survival deficit)
    exp = 1.0 - S.min(axis=1)
    sig = np.stack([np.array([n[1] for n in game.routes[i]], float) for i in range(R)])
    best_blind = np.inf
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
            v = stackM[T].mean(axis=0).max()
            best_blind = min(best_blind, float(v))
    print(f"  strongest payoff-BLIND safest+separated stack found: {best_blind:.4f} "
          f"({best_blind/sol.loss_mixed:.2f}x eq) -> SACRED 0.742 survives blind rules, "
          f"loses to the oracle-fitted class and to tabular FP")


def part2_theatre():
    from src.envs.aerial_theatre_vec import load_vec_theatre, build_theatre_game, _axis
    th = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(
        th, K=1, n_lanes=14, n_terrain=12, spacing_km=2.0, standoff_km=4.0)
    sol = solve_multiconvoy(game, N, "mission")
    _, M = objective_matrix(game, N, "mission", 1)
    R = game.n_routes
    _, oidx = occ_maps(R)
    stackM = stack_rows(M, R, oidx)
    exp = 1.0 - S.min(axis=1)
    rows = {}
    d = np.zeros(R); d[lane_idx] = 1 / len(lane_idx); rows["uniform_LANES (ledger row)"] = stack_val(stackM, d)
    d = np.zeros(R); d[lane_idx] = 1 / np.clip(exp[lane_idx], 1e-9, None); d /= d.sum()
    rows["invrisk_LANES"] = stack_val(stackM, d)
    rows["uniform_FULL"] = stack_val(stackM, np.full(R, 1 / R))
    d = 1 / np.clip(exp, 1e-9, None); rows["invrisk_FULL"] = stack_val(stackM, d / d.sum())
    cover = [i for i in range(R) if i not in lane_idx]
    if cover:
        d = np.zeros(R); d[cover] = 1 / len(cover); rows["uniform_COVER"] = stack_val(stackM, d)
    order = np.argsort(exp)
    for k in (3, 5, 8):
        d = np.zeros(R); d[order[:k]] = 1 / k; rows[f"uniform_top{k}safest"] = stack_val(stackM, d)
    print(f"\n[vec theatre K1 standoff4] R={R} (lanes {len(lane_idx)}) H={len(coords)} "
          f"eq={sol.loss_mixed:.4f} det={sol.loss_det:.4f}")
    for k, v in sorted(rows.items(), key=lambda kv: kv[1]):
        print(f"  {k:28s} {v:.4f} ({v/sol.loss_mixed:.2f}x eq)")
    for k, (v, T) in best_k_stacks(stackM, ks=(2, 3, 4, 5)).items():
        print(f"  best {k}-route uniform stack: {v:.4f} ({v/sol.loss_mixed:.2f}x eq) routes={T}")
    # payoff-blind greedy max-separation on lateral signatures
    u2, nrm2 = _axis(th)

    def sig(rte):
        al = np.array([(np.asarray(p) - th.base) @ u2 for p in rte])
        la = np.array([(np.asarray(p) - th.base) @ nrm2 for p in rte])
        return np.interp(np.linspace(al.min(), al.max(), 10), al, la)
    sigs = np.stack([sig(m) for m in menu])
    for k in (3, 4, 5):
        dmat = np.linalg.norm(sigs[:, None] - sigs[None], axis=2)
        i, j = np.unravel_index(np.argmax(dmat), dmat.shape)
        T = [int(i), int(j)]
        while len(T) < k:
            cand = [p for p in range(R) if p not in T]
            dd = [min(np.linalg.norm(sigs[c] - sigs[t]) for t in T) for c in cand]
            T.append(cand[int(np.argmax(dd))])
        v = stackM[T].mean(axis=0).max()
        print(f"  payoff-BLIND greedy-max-separation k={k}: {v:.4f} ({v/sol.loss_mixed:.2f}x eq)")


if __name__ == "__main__":
    part1_fleet_cell()
    try:
        part2_theatre()
    except FileNotFoundError:
        print("[theatre] data/maps/theatre_kgd_gvardeysk_vec.json not present; part 2 skipped")
