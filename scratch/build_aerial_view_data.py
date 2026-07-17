"""Export the gen28 aerial game data for the interactive visualisation (three K=1 instances:
the A1 headline pinch+banded cell, an open-sector cell, one held-out random layout). Everything
the page needs to reproduce the game faithfully in JS: lattice, menu, hazard grid + field,
strategies (shortest / uniform-lane / inv-risk-lane / equilibrium), the equilibrium attacker
mixture and per-strategy best-response positions, and the oracle values the Monte-Carlo must
converge to. The proximity taper is recomputed in JS from (centres, r, p_max) with the same
linear formula, so the animation IS the payoff model, not a cartoon of it."""

import json

import numpy as np

from src.baselines.aerial_lanes import lane_menu_indices, lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_sector import (SectorLattice, banded_pmax, build_aerial_game,
                                    build_aerial_menu, hazard_grid, lane_path, path_length,
                                    route_survival_matrix)
from scripts.train_aerial_generalist import random_field

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))


def export(name, title, lat, r, pmax_spec):
    menu = build_aerial_menu(lat, R=40)
    centres = hazard_grid(lat)
    if pmax_spec == "banded":
        pm = banded_pmax(centres, lat.ny)
    elif isinstance(pmax_spec, (int, float)):
        pm = np.full(len(centres), float(pmax_spec))
    else:  # layout seed
        pm = random_field(centres, int(pmax_spec[1]))
    game = build_aerial_game(lat, menu, centres, K=1, r=r, p_max=pm)
    S = route_survival_matrix(menu, centres, r=r, p_max=pm)
    sol = solve(game)
    lane_idx = lane_menu_indices(lat, menu, r)
    stacks = lane_stack_distributions(game, lane_idx, S)
    sp = lane_path(lat, lat.base[1])
    shortest = np.zeros(game.n_routes)
    shortest[menu.index(sp) if sp in menu else int(np.argmin(game.travel_cost))] = 1.0
    strategies = {
        "shortest": shortest,
        "uniform_lane": stacks["uniform_lane"],
        "invrisk_lane": stacks["invrisk_lane"],
        "equilibrium": sol.defender_strategy,
    }
    out = dict(
        name=name, title=title, ny=lat.ny, nx=lat.nx,
        blocked=sorted([list(b) for b in lat.blocked]),
        base=list(lat.base), target=list(lat.target), r=r, K=1,
        pmax=[round(float(x), 4) for x in pm],
        centres=[[float(a), float(b)] for a, b in centres],
        routes=[[list(n) for n in p] for p in menu],
        costs=[round(path_length(p), 3) for p in menu],
        lane_idx=lane_idx,
        eq_value=round(float(sol.value), 4), loss_det=round(float(sol.loss_det), 4),
        attacker_eq=[round(float(x), 5) for x in sol.attacker_strategy],
        strategies={}, worst={}, br_pos={},
    )
    for k, d in strategies.items():
        j, v = best_response_attacker(game, d)
        out["strategies"][k] = [round(float(x), 5) for x in d]
        out["worst"][k] = round(float(v), 4)
        out["br_pos"][k] = int(game.interdiction_sets[j][0])   # K=1: iset = (position,)
    return out


data = [
    export("pinch_banded", "Pinch + banded effectiveness (A1 headline; K=1, r=1.6)",
           PINCH, 1.6, "banded"),
    export("base_r08", "Open sector, 6 touching lanes (K=1, r=0.8)", BASE, 0.8, 0.9),
    export("holdout2000", "Held-out random threat layout (A3 test; K=1, r=1.2)",
           BASE, 1.2, ("layout", 2000)),
]
with open("models/runs/gen28_view_data.json", "w") as f:
    json.dump(data, f)
print("wrote models/runs/gen28_view_data.json",
      [f"{d['name']}: eq={d['eq_value']} worst={d['worst']}" for d in data])
