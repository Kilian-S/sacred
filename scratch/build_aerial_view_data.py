"""Export GAME V2 data for the interactive sector view: four instances (the A1 headline
pinch+banded cell, the low-phi open sector, a held-out A3 layout, and a K=2 cell so the
budget axis is visible). Curves are exported as sampled points (decimated x2); the page
recomputes the hazard-rate line integral live (kappa = -ln(1-p_max)/r), so the animation IS
the v2 payoff model. Attacker: per-strategy best-response position set + the equilibrium
attacker's support (positions + weights)."""

import json

import numpy as np

from scripts.train_aerial_generalist import random_field
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_curves import build_curve_menu, build_curved_game, dense_hazard_grid
from src.envs.aerial_sector import SectorLattice, banded_pmax

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))


def export(name, title, lat, r, pmax_spec, K=1):
    menu, lane_idx = build_curve_menu(lat, r, R=40, seed=0)
    centres = dense_hazard_grid(lat, step=0.5)
    if pmax_spec == "banded":
        pm = banded_pmax(centres, lat.ny)
    elif isinstance(pmax_spec, (int, float)):
        pm = np.full(len(centres), float(pmax_spec))
    else:
        pm = random_field(centres, int(pmax_spec[1]))
    game, S = build_curved_game(lat, menu, centres, K, r=r, p_max=pm)
    sol = solve(game)
    stacks = lane_stack_distributions(game, lane_idx, S)
    shortest = np.zeros(game.n_routes)
    shortest[int(np.argmin(game.travel_cost))] = 1.0
    strategies = {"shortest": shortest, "uniform_lane": stacks["uniform_lane"],
                  "invrisk_lane": stacks["invrisk_lane"], "equilibrium": sol.defender_strategy}
    support = [{"p": [int(x) for x in game.interdiction_sets[j]], "w": round(float(w), 5)}
               for j, w in enumerate(sol.attacker_strategy) if w > 1e-4]
    out = dict(
        name=name, title=title, ny=lat.ny, nx=lat.nx,
        blocked=sorted([list(b) for b in lat.blocked]),
        base=list(lat.base), target=list(lat.target), r=r, K=K,
        pmax=[round(float(x), 4) for x in pm],
        centres=[[float(a), float(b)] for a, b in centres],
        routes=[[[round(float(x), 3), round(float(y), 3)] for x, y in c.pts[::2]]
                for c in menu],
        lengths=[round(c.length, 2) for c in menu],
        lane_idx=lane_idx,
        eq_value=round(float(sol.value), 4), loss_det=round(float(sol.loss_det), 4),
        attacker_support=support, strategies={}, worst={}, br_pos={},
    )
    for k, d in strategies.items():
        j, v = best_response_attacker(game, d)
        out["strategies"][k] = [round(float(x), 5) for x in d]
        out["worst"][k] = round(float(v), 4)
        out["br_pos"][k] = [int(x) for x in game.interdiction_sets[j]]
    return out


data = [
    export("pinch_banded", "A1 headline: pinch + banded field (K=1, r=1.2)", PINCH, 1.2, "banded"),
    export("base_r08", "Open sector, low coverage (K=1, r=0.8)", BASE, 0.8, 0.9),
    export("holdout2000", "Held-out random threat layout (A3 test; K=1, r=1.2)",
           BASE, 1.2, ("layout", 2000)),
    export("base_K2", "Two hazards committed (K=2, r=1.2)", BASE, 1.2, 0.9, K=2),
]
with open("models/runs/gen28_view_data.json", "w") as f:
    json.dump(data, f)
print("wrote models/runs/gen28_view_data.json",
      [f"{d['name']}: eq={d['eq_value']} worst={d['worst']}" for d in data])
