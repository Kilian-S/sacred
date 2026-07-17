"""gen28 v2.1 layout-FAMILY probe (oracle-only): which (lattice, r, K) family keeps a material
best-naive margin on RANDOM threat layouts UNDER STANDOFF ZONES? Ran 2026-07-17; numbers in the
ledger's v2.1 amendment. Result: BASE r1.6 K1 is the A3 family (best-naive/eq median 1.34,
min 1.20; robust-static 1.32; cross-play 1.80); pinch families lose layout-dependence (cross
1.28-1.48) because the gap funnels everything."""
import numpy as np
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker, solve, _row_minimiser
from src.envs.aerial_curves import build_curve_menu, build_curved_game, dense_hazard_grid
from src.envs.aerial_sector import SectorLattice
from scripts.train_aerial_generalist import random_field

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset({(6,j) for j in range(9) if j not in (3,4,5)}))

def family(tag, lat, r, K, n=8):
    menu, lane_idx = build_curve_menu(lat, r, R=40, seed=0)
    centres = dense_hazard_grid(lat, step=0.5)
    games, eqs, sols, inv, best = [], [], [], [], []
    for s in range(n):
        pm = random_field(centres, 1000+s)
        game, S = build_curved_game(lat, menu, centres, K, r=r, p_max=pm)
        sol = solve(game); games.append(game); eqs.append(sol.value); sols.append(sol.defender_strategy)
        vals = {k: best_response_attacker(game, v)[1]
                for k, v in lane_stack_distributions(game, lane_idx, S).items()}
        inv.append(vals["invrisk_lane"]/sol.value); best.append(min(vals.values())/sol.value)
    _, rx = _row_minimiser(np.hstack([g.payoff for g in games]))
    rob = [best_response_attacker(g, rx)[1]/e for g, e in zip(games, eqs)]
    cross = [best_response_attacker(games[j], sols[i])[1] for i in range(n) for j in range(n) if i != j]
    print(f"{tag:22s} eq_med={np.median(eqs):.3f} invlane/eq med={np.median(inv):.2f} "
          f"BESTnaive/eq med={np.median(best):.2f} min={min(best):.2f} "
          f"robust/eq med={np.median(rob):.2f} cross/eq={np.mean(cross)/np.mean(eqs):.2f}")

if __name__ == "__main__":
    family("BASE  r1.2 K1", BASE, 1.2, 1)
    family("BASE  r1.6 K1", BASE, 1.6, 1)
    family("PINCH r1.2 K1", PINCH, 1.2, 1)
    family("PINCH r1.6 K1", PINCH, 1.6, 1)
    family("BASE  r1.2 K2", BASE, 1.2, 2, n=6)
    family("PINCH r1.2 K2", PINCH, 1.2, 2, n=6)
