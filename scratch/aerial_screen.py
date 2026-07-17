"""gen28 screen v2 (oracle-only, NO training): the curved-route + line-integral game
(experiments/gen28_aerial.md GAME V2 amendment). Same aiming job as v1 (find where calibrated
mixing beats the strongest naive stack, under the pre-registered baseline set) on the honest
geometry: lanes at CONTINUOUS offsets (the strengthened naive rule), dense 0.5-step hazard
grid at K <= 2 (1.0-step at K = 3, disclosed), and a grid-CONVERGENCE row certifying the
discretisation. v1 (lattice polylines, per-arc Bernoulli, 45-position grid) is retired
pre-training; its numbers remain in the ledger's v1 section and in git history.

Run: OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 PYTHONPATH=. .venv/bin/python \
       scratch/aerial_screen.py --json-out models/runs/gen28_screen.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.baselines.aerial_lanes import lane_stack_distributions, tabular_smooth_fp
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_curves import (build_curve_menu, build_curved_game, dense_hazard_grid,
                                    lane_offsets)
from src.envs.aerial_sector import SectorLattice, banded_pmax, coverage_fraction

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))
R_MENU = 40
TAB_FP_MAX_ISETS = 50_000


def entropy_frac(d: np.ndarray) -> float:
    p = d[d > 1e-12]
    return float(-(p * np.log(p)).sum() / np.log(len(d)))


def screen_cell(lat: SectorLattice, K: int, r: float, *, pmax=0.9, step: float = 0.5,
                tag: str = "") -> dict:
    menu, lane_idx = build_curve_menu(lat, r, R=R_MENU, seed=0)
    centres = dense_hazard_grid(lat, step=step)
    pm = banded_pmax(centres, lat.ny) if pmax == "banded" else float(pmax)
    t0 = time.time()
    game, S = build_curved_game(lat, menu, centres, K, r=r, p_max=pm)
    t_build = time.time() - t0
    t0 = time.time()
    sol = solve(game)
    t_solve = time.time() - t0

    rows: dict[str, float] = {}
    for name, d in lane_stack_distributions(game, lane_idx, S).items():
        _, rows[name] = best_response_attacker(game, d)
    straight = np.zeros(game.n_routes)
    straight[int(np.argmin(game.travel_cost))] = 1.0
    rows["shortest_det"] = float(best_response_attacker(game, straight)[1])

    t_fp = None
    if game.payoff.shape[1] <= TAB_FP_MAX_ISETS:
        t0 = time.time()
        rows["tabular_fp"], _ = tabular_smooth_fp(game, rounds=3000)
        t_fp = round(time.time() - t0, 2)

    naive = [rows[k] for k in ("uniform_lane", "invrisk_lane", "uniform_full", "invrisk_full")
             if k in rows]
    best_naive = min(naive)
    eq = sol.value
    return dict(
        tag=tag, pinch=bool(lat.blocked), K=K, r=r, step=step,
        pmax=("banded" if pmax == "banded" else float(pmax)),
        phi=coverage_fraction(K, r, lat.W), n_routes=game.n_routes,
        n_lanes=len(lane_offsets(lat, r)), n_pos=len(centres),
        n_isets=game.payoff.shape[1],
        eq=eq, loss_det=sol.loss_det,
        leader_entropy_frac=entropy_frac(sol.defender_strategy),
        eq_cost=float(sol.defender_strategy @ game.travel_cost),
        min_cost=float(game.travel_cost.min()),
        best_naive=best_naive,
        best_naive_over_eq=(best_naive / eq if eq > 1e-9 else None),
        det_over_eq=(sol.loss_det / eq if eq > 1e-9 else None),
        t_build=round(t_build, 2), t_solve=round(t_solve, 2), t_fp=t_fp,
        **{k: round(v, 4) for k, v in rows.items()},
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="models/runs/gen28_screen.json")
    args = ap.parse_args()
    cells: list[dict] = []

    def run(name, **kw):
        c = screen_cell(tag=name, **kw)
        cells.append(c)
        print(f"{name:26s} K={c['K']} r={c['r']:.1f} phi={c['phi']:.2f} step={c['step']} "
              f"eq={c['eq']:.3f} det={c['loss_det']:.3f} bestnaive={c['best_naive']:.3f} "
              f"naive/eq={c['best_naive_over_eq']:.2f} H={c['leader_entropy_frac']:.2f} "
              f"lanes={c['n_lanes']} (t={c['t_build']:.1f}+{c['t_solve']:.1f}s)", flush=True)

    for K in (1, 2):
        for r in (0.8, 1.2, 1.6, 2.0):
            run(f"base_K{K}_r{r}", lat=BASE, K=K, r=r)
    for r in (0.8, 1.2, 1.6):
        run(f"base_K3_r{r}", lat=BASE, K=3, r=r, step=1.0)     # K=3: coarse grid, disclosed
    for K in (1, 2):
        for r in (1.2, 1.6, 2.0):
            run(f"banded_K{K}_r{r}", lat=BASE, K=K, r=r, pmax="banded")
    for K in (1, 2):
        for r in (1.2, 1.6):
            run(f"pinch_K{K}_r{r}", lat=PINCH, K=K, r=r)
            run(f"pinch_banded_K{K}_r{r}", lat=PINCH, K=K, r=r, pmax="banded")
    # grid-convergence certificate (the discretised-adversary defence)
    for step in (1.0, 0.5, 0.25):
        run(f"conv_K1_r1.2_s{step}", lat=BASE, K=1, r=1.2, step=step)
    for step in (1.0, 0.5):
        run(f"conv_K2_r1.2_s{step}", lat=BASE, K=2, r=1.2, step=step)

    with open(args.json_out, "w") as f:
        json.dump({"version": "v2-curved-integral", "cells": cells}, f, indent=1)

    ok = [c for c in cells if not c["tag"].startswith("conv")
          and c["best_naive_over_eq"] and 0.02 < c["eq"] < 0.9
          and c["leader_entropy_frac"] < 0.9]
    ok.sort(key=lambda c: -c["best_naive_over_eq"])
    print("\n=== SHORTLIST (naive-gap ranked; gates: 0.02<eq<0.9, H/lnR<0.9) ===")
    for c in ok[:12]:
        print(f"{c['tag']:26s} phi={c['phi']:.2f} eq={c['eq']:.3f} "
              f"naive/eq={c['best_naive_over_eq']:.2f} det/eq={c['det_over_eq']:.2f} "
              f"H={c['leader_entropy_frac']:.2f} tabFP={c.get('tabular_fp')}")
    conv = [c for c in cells if c["tag"].startswith("conv")]
    print("\n=== GRID CONVERGENCE (eq vs step) ===")
    for c in conv:
        print(f"{c['tag']:26s} step={c['step']} pos={c['n_pos']} eq={c['eq']:.4f} "
              f"best_naive={c['best_naive']:.4f}")
    print(f"\n{len(cells)} cells -> {args.json_out}")


if __name__ == "__main__":
    main()
