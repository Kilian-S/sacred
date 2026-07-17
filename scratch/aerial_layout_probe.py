"""gen28 A3 aiming probe (oracle-only): do random hidden-hazard THREAT FIELDS (per-position
effectiveness layouts, spatially correlated, decorrelated from lattice geometry by construction)
vary enough that (a) an unconditioned static mixture cannot cover them (the robust-static cap),
and (b) the two-line LAYOUT-AWARE naive rule (inverse-risk lane stack, re-weighted per layout)
stays materially suboptimal? Also: menu-sufficiency at the two headline-candidate cells.

Rows per layout (all oracle-exact): eq | uniform-lane | inv-risk-lane (layout-aware) |
inv-risk-full (layout-aware) | robust-static (one mixture, worst-case over ALL layouts) |
cross-play mean (another layout's equilibrium played here). A3 is aimable if
invrisk_lane/eq and robust/eq are materially > 1 on most layouts.
"""

from __future__ import annotations

import json

import numpy as np

from src.baselines.aerial_lanes import lane_menu_indices, lane_stack_distributions
from src.baselines.interdiction_oracle import (InterdictionGame, best_response_attacker,
                                               solve, _row_minimiser)
from src.envs.aerial_sector import (SectorLattice, build_aerial_game, build_aerial_menu,
                                    hazard_grid, route_survival_matrix)

LAT = SectorLattice(ny=9, nx=13)
K, R_HAZ = 1, 1.2
N_LAYOUTS = 12
BAND = (0.30, 0.95)


def random_field(centres: np.ndarray, seed: int, length_scale: float = 2.5) -> np.ndarray:
    """Spatially correlated effectiveness field over candidate positions: an RBF-kernel Gaussian
    draw mapped affinely into BAND by rank (a smooth 'danger region' intel picture, independent
    of the lattice geometry)."""
    rng = np.random.default_rng(seed)
    d2 = ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
    cov = np.exp(-d2 / (2.0 * length_scale ** 2)) + 1e-8 * np.eye(len(centres))
    g = rng.multivariate_normal(np.zeros(len(centres)), cov)
    ranks = np.argsort(np.argsort(g))
    lo, hi = BAND
    return lo + (hi - lo) * ranks / (len(centres) - 1)


def main() -> None:
    menu = build_aerial_menu(LAT, R=40)
    centres = hazard_grid(LAT)
    lane_idx = lane_menu_indices(LAT, menu, R_HAZ)

    games, eqs, sols, rows = [], [], [], []
    for s in range(N_LAYOUTS):
        pm = random_field(centres, seed=1000 + s)
        game = build_aerial_game(LAT, menu, centres, K, r=R_HAZ, p_max=pm)
        S = route_survival_matrix(menu, centres, r=R_HAZ, p_max=pm)
        sol = solve(game)
        d = lane_stack_distributions(game, lane_idx, S)
        row = {k: best_response_attacker(game, v)[1] for k, v in d.items()}
        games.append(game); eqs.append(sol.value); sols.append(sol.defender_strategy)
        rows.append(row)

    # robust-static: ONE mixture minimising worst-case interception over the union of all
    # layouts' interdiction columns = the best any unconditioned object can guarantee.
    stacked = np.hstack([g.payoff for g in games])
    robust_worst, robust_x = _row_minimiser(stacked)
    robust_per_layout = [best_response_attacker(g, robust_x)[1] for g in games]

    # cross-play: layout i's equilibrium strategy scored under layout j's best response.
    cross = [best_response_attacker(games[j], sols[i])[1]
             for i in range(N_LAYOUTS) for j in range(N_LAYOUTS) if i != j]
    own = float(np.mean(eqs))

    out = dict(
        n_layouts=N_LAYOUTS, K=K, r=R_HAZ, band=BAND,
        eq_mean=own, eq_range=[float(min(eqs)), float(max(eqs))],
        uniform_lane_over_eq=[rows[s]["uniform_lane"] / eqs[s] for s in range(N_LAYOUTS)],
        invrisk_lane_over_eq=[rows[s]["invrisk_lane"] / eqs[s] for s in range(N_LAYOUTS)],
        invrisk_full_over_eq=[rows[s]["invrisk_full"] / eqs[s] for s in range(N_LAYOUTS)],
        robust_static_over_eq=[robust_per_layout[s] / eqs[s] for s in range(N_LAYOUTS)],
        crossplay_mean_over_eq=float(np.mean(cross)) / own,
    )
    for k in ("uniform_lane_over_eq", "invrisk_lane_over_eq", "invrisk_full_over_eq",
              "robust_static_over_eq"):
        v = np.asarray(out[k])
        print(f"{k:26s} median={np.median(v):.2f} min={v.min():.2f} max={v.max():.2f}")
    print(f"crossplay_mean_over_eq     {out['crossplay_mean_over_eq']:.2f}")
    print(f"eq mean {own:.3f} range {out['eq_range']}")

    # menu sufficiency at the two headline-candidate cells
    print("\nmenu sufficiency (eq vs R):")
    for tag, lat, pmax in (("pinch_banded_K1_r1.6",
                            SectorLattice(ny=9, nx=13, blocked=frozenset(
                                {(6, j) for j in range(9) if j not in (3, 4, 5)})), "banded"),
                           ("base_K1_r0.8", LAT, 0.9)):
        from src.envs.aerial_sector import banded_pmax
        cs = hazard_grid(lat)
        pm = banded_pmax(cs, lat.ny) if pmax == "banded" else pmax
        rr = 1.6 if "1.6" in tag else 0.8
        vals = []
        for R in (20, 40, 60, 80):
            m = build_aerial_menu(lat, R=R)
            g = build_aerial_game(lat, m, cs, 1, r=rr, p_max=pm)
            vals.append(round(solve(g).value, 4))
        print(f"  {tag}: R=20/40/60/80 -> {vals}")

    with open("models/runs/gen28_layout_probe.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n-> models/runs/gen28_layout_probe.json")


if __name__ == "__main__":
    main()
