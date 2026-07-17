"""gen28 v3.0 FLEET screen (oracle-only, free): the road-proven register transplanted - N=3
UAVs, fleet menu-select, loss-averse MISSION objective P(>=1 intercepted) - on the aerial
curved game (v2.2 geometry, standoff zones). Complete naive family at N=3: STACK rules
(fleet on one sampled route: lane sets per spacing + full menu, uniform + inverse-risk) AND
INDEPENDENT-mixing rules (each UAV samples independently), + det (= the ALNS-class optimum)
+ equilibrium. Kilian's 2026-07-17 night mandate: re-aim where supremacy is honest."""
import itertools
import numpy as np
from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker
from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy, _row_minimiser
from src.envs.aerial_curves import (all_lane_sets, build_curve_menu, build_curved_game,
                                    dense_hazard_grid)
from src.envs.aerial_sector import SectorLattice, banded_pmax
from scripts.train_aerial_generalist import random_field

BASE = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset({(6,j) for j in range(9) if j not in (3,4,5)}))
DBL = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(4,j) for j in range(9) if j < 5} | {(8,j) for j in range(9) if j > 3}))
N = 3

def occ_maps(R):
    occs = list(itertools.combinations_with_replacement(range(R), N))
    vecs = np.zeros((len(occs), R))
    for i, c in enumerate(occs):
        for r in c: vecs[i, r] += 1
    idx = {tuple(int(x) for x in v): i for i, v in enumerate(vecs)}
    return occs, vecs, idx

def stack_occ_dist(d_routes, vecs, idx):
    out = np.zeros(len(vecs))
    for r, p in enumerate(d_routes):
        if p > 1e-12:
            v = np.zeros(d_routes.shape[0]); v[r] = N
            out[idx[tuple(int(x) for x in v)]] += p
    return out

def indep_occ_dist(d_routes, occs, vecs):
    from math import factorial
    out = np.zeros(len(vecs))
    for i, c in enumerate(occs):
        counts = {}
        for r in c: counts[r] = counts.get(r, 0) + 1
        coef = factorial(N)
        p = 1.0
        for r, k in counts.items():
            coef //= factorial(k); p *= d_routes[r] ** k
        out[i] = coef * p
    return out

def screen(tag, lat, r_menu, pmax, K=1, do_fp=False):
    menu, _ = build_curve_menu(lat, r_menu, R=40, seed=0)
    centres = dense_hazard_grid(lat, step=0.5)
    pm = banded_pmax(centres, lat.ny) if pmax == "banded" else (
        pmax if not isinstance(pmax, tuple) else random_field(centres, pmax[1]))
    game, S = build_curved_game(lat, menu, centres, K, r=r_menu, p_max=pm)
    sol = solve_multiconvoy(game, N, "mission")
    occs, vecs, idx = occ_maps(game.n_routes)
    _, M = objective_matrix(game, N, "mission", 1)
    def val(occ_dist): return float((occ_dist @ M).max())
    rows = {}
    lsets = all_lane_sets(lat, menu)
    for rc, li in (lsets.items() if lsets else [(0.0, [])]):
        for k, d in lane_stack_distributions(game, li, S).items():
            rows[f"{k}@{rc}|stack"] = val(stack_occ_dist(d, vecs, idx))
            if k in ("uniform_lane", "uniform_full"):
                rows[f"{k}@{rc}|indep"] = val(indep_occ_dist(d, occs, vecs))
    bn = min(rows.values())
    fp = None
    if do_fp:
        x = np.full(len(vecs), 1.0/len(vecs)); tot = np.zeros(len(vecs))
        for t in range(1, 1501):
            tot += x; j = int(((tot/t) @ M).argmax())
            x = x*np.exp(-0.25*M[:, j]); x /= x.sum()
        fp = val(tot/1500)
    H = -(sol.defender_strategy[sol.defender_strategy>1e-9]
          * np.log(sol.defender_strategy[sol.defender_strategy>1e-9])).sum()/np.log(len(vecs))
    print(f"{tag:28s} eq={sol.loss_mixed:.3f} det={sol.loss_det:.3f} bestnaive={bn:.3f} "
          f"naive/eq={bn/sol.loss_mixed:.2f} det/eq={sol.loss_det/sol.loss_mixed:.2f} "
          f"H={H:.2f}" + (f" tabFP={fp:.3f}" if fp else ""), flush=True)
    return sol.loss_mixed, bn

if __name__ == "__main__":
    screen("FLEET dblpinch_banded_r1.2", DBL, 1.2, "banded", do_fp=True)
    screen("FLEET pinch_banded_r1.6", PINCH, 1.6, "banded", do_fp=True)
    screen("FLEET banded_r1.6", BASE, 1.6, "banded", do_fp=True)
    screen("FLEET base_r1.2", BASE, 1.2, 0.9)
    screen("FLEET base_r0.8", BASE, 0.8, 0.9)
    for fam, lat, rr, s0 in (("B", BASE, 1.6, 2000), ("D", DBL, 1.2, 2100)):
        vals = [screen(f"FLEET layout{fam}{s0+s}", lat, rr, ("layout", s0+s)) for s in range(3)]
        print(f"   family {fam}: naive/eq " +
              " ".join(f"{b/e:.2f}" for e, b in vals), flush=True)
