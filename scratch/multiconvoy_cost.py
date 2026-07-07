"""Probe C: the travel-cost frontier. The deterministic coordinator trades fleet travel-cost against
mission-failure risk (a genuine multi-objective VRP an ALNS solves -> Obj-5 non-degenerate), and
SACRED's randomised frontier DOMINATES it (lower risk at the same cost). Multi-convoy, mission obj,
soft interception. NO training."""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from scratch.multiconvoy_probe import occupancies, row_minimiser
from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from scratch.vuln_band_probe import kaliningrad


def cost_constrained_mixed(M, costs, budget):
    """min v s.t. M^T x <= v ; costs.x <= budget ; sum x = 1 ; x>=0."""
    n, m = M.shape
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub = np.vstack([np.hstack([M.T, -np.ones((m, 1))]),
                      np.hstack([costs[None, :], np.zeros((1, 1))])])
    b_ub = np.concatenate([np.zeros(m), [budget]])
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return float(res.x[-1]) if res.success else float("nan")


def main(s="110", t="135", N=2, K=1, band=(0.15, 0.95)):
    G = kaliningrad()
    routes = build_route_set(G, str(s), str(t), 0, "w")
    cand = set().union(*(edges_of_route(r) for r in routes))
    ifn = survival_intercept_fn(length_band_vulnerability(G, cand, band=band, weight="w"))
    game = build_interdiction_game(G, str(s), str(t), K=K, k_extra=0, intercept_fn=ifn)
    R = game.n_routes
    occs = list(occupancies(R, N))
    # mission-failure matrix + per-strategy fleet travel-cost
    M = np.zeros((len(occs), game.payoff.shape[1]))
    fleet_cost = np.zeros(len(occs))
    for oi, o in enumerate(occs):
        fleet_cost[oi] = float(np.asarray(o) @ game.travel_cost)
        for j in range(game.payoff.shape[1]):
            M[oi, j] = 1.0 - float(np.prod((1.0 - game.payoff[:, j]) ** o))
    cmin, cmax = fleet_cost.min(), fleet_cost.max()
    print(f"{s}->{t} soft, N={N} convoys, K={K}. Fleet travel-cost range [{cmin:.1f}, {cmax:.1f}].")
    print("Mission-failure risk vs cost budget:  DETERMINISTIC (ALNS) | MIXED (SACRED)")
    print(f"  {'budget':>7} {'det_risk':>9} {'mixed_risk':>11} {'gap':>6}")
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        budget = cmin + frac * (cmax - cmin)
        feas = M[fleet_cost <= budget + 1e-9]
        det = float(min(feas[i].max() for i in range(feas.shape[0]))) if len(feas) else float("nan")
        mix = cost_constrained_mixed(M, fleet_cost, budget)
        print(f"  {budget:7.1f} {det:9.3f} {mix:11.3f} {det-mix:6.3f}")
    print("(cheap budget = must use the short/exposed routes; SACRED still spreads risk by randomising)")


if __name__ == "__main__":
    main("110", "135", N=2, K=1)
    print()
    main("33", "71", N=3, K=1)
