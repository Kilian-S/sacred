"""gen28 v2.2 STRUCTURE probe (oracle-only, free): which realism-flavoured structure widens
the naive-vs-equilibrium gap under standoff zones? Families: staggered double pinch (forces
S-turns), mixed threat radii (large AD sites + small ambush teams), and their combinations
with banded/random fields. Kilian's directive 2026-07-17: shift advantage to SACRED through
honest structure, never through firepower (the screen already shows K/r compress the gap)."""
import numpy as np
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_curves import (build_curve_menu, build_curved_game, dense_hazard_grid,
                                    lane_offsets)
from src.envs.aerial_sector import SectorLattice, banded_pmax
from scripts.train_aerial_generalist import random_field

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset({(6,j) for j in range(9) if j not in (3,4,5)}))
# staggered double pinch: wall at x=4 open TOP (rows 5-8), wall at x=8 open BOTTOM (rows 0-3)
DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4,j) for j in range(9) if j < 5} | {(8,j) for j in range(9) if j > 3}))

def ent(d):
    p = d[d>1e-12]; return float(-(p*np.log(p)).sum()/np.log(len(d)))

def cell(tag, lat, K, r, field, rmix=None):
    menu, lane_idx = build_curve_menu(lat, max(np.atleast_1d(r).max(), 1.2) if rmix else r,
                                      R=40, seed=0)
    if not menu or len(menu) < 10:
        print(f"{tag:34s} menu too small ({len(menu)})"); return
    centres = dense_hazard_grid(lat, step=0.5)
    rng = np.random.default_rng(7)
    rr = (np.where(rng.random(len(centres)) < 0.3, 2.0, 0.8) if rmix else r)
    pm = (banded_pmax(centres, lat.ny) if field == "banded"
          else random_field(centres, 1000) if field == "layout" else 0.9)
    game, S = build_curved_game(lat, menu, centres, K, r=rr, p_max=pm)
    sol = solve(game)
    vals = {k: best_response_attacker(game, v)[1]
            for k, v in lane_stack_distributions(game, lane_idx, S).items()}
    bn = min(vals.values())
    print(f"{tag:34s} eq={sol.value:.3f} det={sol.loss_det:.3f} bestnaive={bn:.3f} "
          f"naive/eq={bn/sol.value:.2f} H={ent(sol.defender_strategy):.2f} "
          f"lanes={len(lane_idx)} R={game.n_routes}")

if __name__ == "__main__":
    print("--- single pinch reference (v2.1 screen values reproduce) ---")
    cell("pinch_banded_K1_r1.6", PINCH, 1, 1.6, "banded")
    print("--- staggered DOUBLE pinch (S-turn forced) ---")
    for K in (1, 2):
        for field in ("uniform", "banded", "layout"):
            cell(f"dblpinch_{field}_K{K}_r1.2", DBL, K, 1.2, field)
    cell("dblpinch_banded_K1_r1.6", DBL, 1, 1.6, "banded")
    print("--- mixed threat radii (30% big r=2.0, 70% small r=0.8), open sector ---")
    for K in (1, 2):
        for field in ("uniform", "banded", "layout"):
            cell(f"mixr_{field}_K{K}", BASE, K, 1.2, field, rmix=True)
    print("--- mixed radii + double pinch ---")
    cell("dbl_mixr_banded_K1", DBL, 1, 1.2, "banded", rmix=True)
    cell("dbl_mixr_layout_K1", DBL, 1, 1.2, "layout", rmix=True)
