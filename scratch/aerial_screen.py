"""gen28 aerial screen (oracle-only, NO training): map the phi = 2Kr/W boundary and find the
cells where calibrated mixing beats the strongest naive lane/menu stack by a material factor,
under the pre-registered baseline set (experiments/gen28_aerial.md). Emits one JSON row per
cell with every baseline, the equilibrium, leader entropy, and solve wall-times (the timing
dogma), plus a ranked shortlist.

Run: OMP_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4 PYTHONPATH=. .venv/bin/python \
       scratch/aerial_screen.py --json-out models/runs/gen28_screen.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.baselines.aerial_lanes import (lane_menu_indices, lane_rows,
                                        lane_stack_distributions, tabular_smooth_fp)
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_sector import (SectorLattice, banded_pmax, build_aerial_game,
                                    build_aerial_menu, coverage_fraction, hazard_grid,
                                    lane_path, route_survival_matrix, solve_cost_weighted)

BASE = SectorLattice(ny=9, nx=13)
# pinch: a wall at column 6 with a 3-row gap (rows 3-5): lane counts vary along the path.
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))
WEATHER = [((5.0, 5.0), 2.0, 3.0), ((8.0, 2.0), 1.5, 2.0)]   # asymmetric observable cells
R_MENU = 40
TAB_FP_MAX_ISETS = 20_000        # tabular-FP row only where the payoff matvec stays cheap


def entropy_frac(d: np.ndarray) -> float:
    p = d[d > 1e-12]
    return float(-(p * np.log(p)).sum() / np.log(len(d)))


def screen_cell(lat: SectorLattice, K: int, r: float, *, taper: str = "linear",
                pmax=0.9, weather=None, lam: float = 0.0, tag: str = "") -> dict:
    menu = build_aerial_menu(lat, R=R_MENU)
    centres = hazard_grid(lat)
    pm = banded_pmax(centres, lat.ny) if pmax == "banded" else float(pmax)
    t0 = time.time()
    game = build_aerial_game(lat, menu, centres, K, r=r, p_max=pm, taper=taper,
                             weather=weather or [])
    t_build = time.time() - t0
    S = route_survival_matrix(menu, centres, r=r, p_max=pm, taper=taper)
    t0 = time.time()
    sol = solve(game)
    t_solve = time.time() - t0

    lane_idx = lane_menu_indices(lat, menu, r)
    rows: dict[str, float] = {}
    dists = lane_stack_distributions(game, lane_idx, S)
    for name, d in dists.items():
        _, rows[name] = best_response_attacker(game, d)
    # shortest-path deterministic = the straight centre lane (the operational default)
    sp = lane_path(lat, lat.base[1])
    sp_idx = menu.index(sp) if sp in menu else int(np.argmin(game.travel_cost))
    rows["shortest_det"] = float(game.payoff[sp_idx].max())

    n_isets = game.payoff.shape[1]
    if n_isets <= TAB_FP_MAX_ISETS:
        t0 = time.time()
        rows["tabular_fp"], _ = tabular_smooth_fp(game, rounds=3000)
        t_fp = time.time() - t0
    else:
        t_fp = None

    if lam > 0.0:
        worst, cost, _ = solve_cost_weighted(game, lam)
        rows["costweighted_worst"] = worst
        rows["costweighted_cost_premium"] = cost / float(game.travel_cost.min()) - 1.0

    naive = [v for k, v in rows.items()
             if k in ("uniform_lane", "invrisk_lane", "uniform_full", "invrisk_full")]
    best_naive = min(naive)
    eq = sol.value
    cell = dict(
        tag=tag, ny=lat.ny, nx=lat.nx, pinch=bool(lat.blocked), K=K, r=r, taper=taper,
        pmax=("banded" if pmax == "banded" else float(pmax)), lam=lam,
        weather=bool(weather), phi=coverage_fraction(K, r, lat.W),
        n_routes=game.n_routes, n_lanes=len(lane_rows(lat, r)), n_isets=n_isets,
        eq=eq, loss_det=sol.loss_det, leader_entropy_frac=entropy_frac(sol.defender_strategy),
        eq_cost=float(sol.defender_strategy @ game.travel_cost),
        min_cost=float(game.travel_cost.min()),
        best_naive=best_naive,
        best_naive_over_eq=(best_naive / eq if eq > 1e-9 else None),
        det_over_eq=(sol.loss_det / eq if eq > 1e-9 else None),
        t_build=round(t_build, 3), t_solve=round(t_solve, 3), t_fp=t_fp,
        **{k: round(v, 4) for k, v in rows.items()},
    )
    return cell


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", default="models/runs/gen28_screen.json")
    args = ap.parse_args()

    cells: list[dict] = []

    def run(name, **kw):
        c = screen_cell(tag=name, **kw)
        cells.append(c)
        bn = c["best_naive_over_eq"]
        print(f"{name:34s} K={c['K']} r={c['r']:.1f} phi={c['phi']:.2f} "
              f"eq={c['eq']:.3f} det={c['loss_det']:.3f} bestnaive={c['best_naive']:.3f} "
              f"naive/eq={bn:.2f} H={c['leader_entropy_frac']:.2f} "
              f"(t={c['t_build']:.1f}+{c['t_solve']:.1f}s)" if bn else f"{name}: degenerate")

    # 1) the phi grid, symmetric base sector, linear taper
    for K in (1, 2, 3, 4):
        for r in (0.8, 1.2, 1.6, 2.0):
            if K == 4 and r >= 1.6:
                continue                      # phi > 1.5: fully saturated, skip the exact build
            run(f"base_K{K}_r{r}", lat=BASE, K=K, r=r)

    # 2) heterogeneous hazard effectiveness (game-side asymmetry, the F1 gate axis)
    for K in (1, 2, 3):
        for r in (1.2, 1.6, 2.0):
            run(f"banded_K{K}_r{r}", lat=BASE, K=K, r=r, pmax="banded")

    # 3) pinch geometry (lane counts vary along the path)
    for K in (1, 2, 3):
        for r in (1.2, 1.6):
            run(f"pinch_K{K}_r{r}", lat=PINCH, K=K, r=r)
            run(f"pinch_banded_K{K}_r{r}", lat=PINCH, K=K, r=r, pmax="banded")

    # 4) Gaussian taper sensitivity
    for K in (1, 2, 3):
        run(f"gauss_K{K}_r1.6", lat=BASE, K=K, r=1.6, taper="gauss")

    # 5) detour-cost weight (+ observable weather) on a mid cell
    for lam in (0.01, 0.03):
        run(f"costw{lam}_K2_r1.6", lat=BASE, K=2, r=1.6, lam=lam)
        run(f"costw{lam}_weather_K2_r1.6", lat=BASE, K=2, r=1.6, lam=lam, weather=WEATHER)

    with open(args.json_out, "w") as f:
        json.dump(cells, f, indent=1)

    # shortlist: material naive gap + non-degenerate + asymmetric enough to train
    ok = [c for c in cells
          if c["best_naive_over_eq"] and 0.02 < c["eq"] < 0.9
          and c["leader_entropy_frac"] < 0.9]
    ok.sort(key=lambda c: -c["best_naive_over_eq"])
    print("\n=== SHORTLIST (naive-gap ranked; gates: 0.02<eq<0.9, H/lnR<0.9) ===")
    for c in ok[:12]:
        print(f"{c['tag']:34s} phi={c['phi']:.2f} eq={c['eq']:.3f} "
              f"naive/eq={c['best_naive_over_eq']:.2f} det/eq={c['det_over_eq']:.2f} "
              f"H={c['leader_entropy_frac']:.2f} tabFP={c.get('tabular_fp')}")
    print(f"\n{len(cells)} cells -> {args.json_out}")


if __name__ == "__main__":
    main()
