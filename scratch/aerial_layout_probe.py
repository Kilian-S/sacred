"""gen28 A3 aiming probe v2 (curved+integral game; oracle-only): do random hidden-hazard THREAT FIELDS (per-position
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

from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import (best_response_attacker, solve, _row_minimiser)
from src.envs.aerial_curves import build_curve_menu, build_curved_game, dense_hazard_grid
from src.envs.aerial_sector import SectorLattice

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
    menu, lane_idx = build_curve_menu(LAT, r=R_HAZ, R=40, seed=0)
    centres = dense_hazard_grid(LAT, step=0.5)

    games, eqs, sols, rows = [], [], [], []
    for s in range(N_LAYOUTS):
        pm = random_field(centres, seed=1000 + s)
        game, S = build_curved_game(LAT, menu, centres, K, r=R_HAZ, p_max=pm)
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

    # menu sufficiency at the two v2 headline-candidate cells (curved menus)
    print("\nmenu sufficiency (eq vs R, curved):")
    for tag, lat, pmax, rr in (("pinch_banded_K1_r1.2",
                                SectorLattice(ny=9, nx=13, blocked=frozenset(
                                    {(6, j) for j in range(9) if j not in (3, 4, 5)})),
                                "banded", 1.2),
                               ("base_K1_r1.2", LAT, 0.9, 1.2)):
        from src.envs.aerial_sector import banded_pmax
        cs = dense_hazard_grid(lat, step=0.5)
        pm = banded_pmax(cs, lat.ny) if pmax == "banded" else pmax
        vals = []
        for R in (20, 40, 60, 80):
            m, _ = build_curve_menu(lat, rr, R=R, seed=0)
            g, _S = build_curved_game(lat, m, cs, 1, r=rr, p_max=pm)
            vals.append(round(solve(g).value, 4))
        print(f"  {tag}: R=20/40/60/80 -> {vals}")

    with open("models/runs/gen28_layout_probe.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n-> models/runs/gen28_layout_probe.json")


if __name__ == "__main__":
    main()
