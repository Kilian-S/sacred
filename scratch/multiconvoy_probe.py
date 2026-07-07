"""Multi-convoy interdiction oracle probe (NO training): is SACRED still winnable, and does it give a
metaheuristic a non-degenerate problem (Obj 5), when N convoys route base->FOB against a committed
K-asset interdictor?

Single-convoy result (why we're here): deterministic route 100% intercepted (loss_det=1.0), mixed
route -> loss_mixed (0.167 on 33->71 K=1). Big gap = SACRED wins, but a metaheuristic on ONE route
degenerates to shortest-path (Obj-5 gap).

Multi-convoy generalises the oracle: the defender chooses an OCCUPANCY of N interchangeable convoys
over the candidate routes (a joint routing = a coordination problem a metaheuristic can actually
solve); the attacker commits K interdiction edges (shared across convoys, the hidden Stackelberg
commit). We compute loss_det (best DETERMINISTIC joint plan, worst-cased = what ALNS produces),
loss_mixed (minimax randomised joint strategy = what SACRED targets), and the gap, under TWO
objectives that bracket reality:
  * linear    = expected FRACTION of convoys intercepted (risk-neutral);
  * mission   = P(>= 1 convoy intercepted) = mission-failure prob (loss-averse; realistic for
                contested resupply where losing any convoy is a failure).
The key question this answers that single-convoy could not: does DETERMINISTIC coordination
(spreading convoys across routes, which ALNS does) already capture the benefit that mixing gave in
single-convoy (which would SHRINK SACRED's edge), or does randomised coordination still win big
(which KEEPS SACRED's edge while making the metaheuristic non-degenerate)?

Run: PYTHONPATH=. .venv/bin/python scratch/multiconvoy_probe.py
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

from src.baselines.interdiction_oracle import (
    build_interdiction_game, build_route_set, edges_of_route,
    length_band_vulnerability, survival_intercept_fn)
from scratch.vuln_band_probe import kaliningrad


def row_minimiser(M: np.ndarray) -> tuple[float, np.ndarray]:
    """Zero-sum matrix game: ROW minimises, COL maximises. Returns (value, row mixed strategy)."""
    n, m = M.shape
    c = np.zeros(n + 1); c[-1] = 1.0
    A_ub = np.hstack([M.T, -np.ones((m, 1))]); b_ub = np.zeros(m)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0; b_eq = np.array([1.0])
    bounds = [(0.0, 1.0)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        raise RuntimeError(res.message)
    x = np.clip(res.x[:n], 0.0, None); x = x / x.sum()
    return float(res.x[-1]), x


def occupancies(R: int, N: int):
    """All ways to place N interchangeable convoys over R routes (occupancy vectors summing to N)."""
    for combo in itertools.combinations_with_replacement(range(R), N):
        vec = np.zeros(R, dtype=int)
        for r in combo:
            vec[r] += 1
        yield vec


def multiconvoy_matrix(payoff: np.ndarray, N: int, objective: str) -> tuple[list[np.ndarray], np.ndarray]:
    """Defender occupancy strategies x attacker interdiction sets -> loss matrix.
    payoff[r, j] = interception PROB of route r under interdiction set j (0/1 hard, (0,1) soft)."""
    R, n_isets = payoff.shape
    occs = list(occupancies(R, N))
    M = np.zeros((len(occs), n_isets))
    for oi, o in enumerate(occs):
        for j in range(n_isets):
            p = payoff[:, j]
            if objective == "linear":
                M[oi, j] = float(o @ p) / N                       # expected fraction caught
            else:  # mission: P(>=1 caught), convoys on a route independently Bernoulli(p_r)
                M[oi, j] = 1.0 - float(np.prod((1.0 - p) ** o))
    return occs, M


def analyse(label: str, od, k_extra: int, band, Ns=(1, 2, 3), Ks=(1, 2)):
    G = kaliningrad(); s, t = od
    ifn = None
    if band is not None:
        routes = build_route_set(G, s, t, k_extra, "w")
        cand = set().union(*(edges_of_route(r) for r in routes))
        ifn = survival_intercept_fn(length_band_vulnerability(G, cand, band=band, weight="w"))
    print(f"\n================  {label}  ================")
    for K in Ks:
        game = build_interdiction_game(G, s, t, K=K, k_extra=k_extra, intercept_fn=ifn)
        R, n_isets = game.payoff.shape
        print(f"  routes={R}  interdiction-sets(K={K})={n_isets}")
        print(f"  {'obj':7s} {'N':>2} {'loss_det':>9} {'loss_mixed':>10} {'gap':>6}  {'det-plan(#routes used)':>22}")
        for objective in ("linear", "mission"):
            for N in Ns:
                occs, M = multiconvoy_matrix(game.payoff, N, objective)
                det_i = int(np.argmin(M.max(axis=1)))
                loss_det = float(M[det_i].max())
                loss_mixed, _ = row_minimiser(M)
                spread = int((occs[det_i] > 0).sum())
                print(f"  {objective:7s} {N:>2} {loss_det:9.3f} {loss_mixed:10.3f} "
                      f"{loss_det - loss_mixed:6.3f}  {spread:>3d} of {R} routes "
                      f"({'STACK' if spread == 1 else 'SPREAD'})")
            print()


if __name__ == "__main__":
    print("Multi-convoy interdiction oracle: loss_det (deterministic/ALNS) vs loss_mixed (SACRED).")
    print("gap large => SACRED wins; det-plan SPREAD => a metaheuristic has a real coordination job.")
    analyse("DISJOINT 33->71 (6 routes, hard)", ("33", "71"), 0, None)
    analyse("SHARED-EDGE 33->71 k8 (11 routes, hard)", ("33", "71"), 8, None)
    analyse("BAND 110->135 (3 routes, soft)", ("110", "135"), 0, (0.15, 0.95))
