#!/usr/bin/env python3
"""Oracle-only probe (no training): the MISSING naive-randomisation rows for the multi-convoy
ladders, plus a route-menu sufficiency check.

Motivation (CRITIQUE_EXAMINER.md, 2026-07-12): the multi-convoy ladders report shortest / vanilla /
ALNS-forced-stack / ALNS / SACRED / equilibrium but no UNIFORM row, unlike the single-convoy ladder
(uniform 0.455). Under a best-response metric any deterministic plan is maximally exploited, so the
sharpest sceptical question is not "does SACRED beat ALNS?" but "does SACRED beat the 3-line
heuristic: stack the fleet on ONE uniformly-random route?" This probe measures that row, the
uniform-INDEPENDENT row (each convoy uniform on its own), and how the equilibrium value moves as the
candidate-route menu grows (is the k_extra=8 menu-relative equilibrium an artefact of menu size?).

Everything is exact oracle arithmetic (LP + matrix), seconds, zero training.
"""
from __future__ import annotations

import math

import numpy as np

from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_matrix, solve_multiconvoy)
from src.envs.multiconvoy_interdiction import make_multiconvoy_env


def exploitability(obj_matrix: np.ndarray, dist: np.ndarray) -> float:
    _, v = best_response_attacker_multi(obj_matrix, dist)
    return float(v)


def uniform_stack_dist(occs: list[np.ndarray], N: int) -> np.ndarray:
    """All N convoys on ONE route, route drawn uniformly."""
    dist = np.zeros(len(occs))
    stacked = [i for i, o in enumerate(occs) if int(o.max()) == N]
    for i in stacked:
        dist[i] = 1.0 / len(stacked)
    return dist


def uniform_independent_dist(occs: list[np.ndarray], N: int, R: int) -> np.ndarray:
    """Each convoy picks a route independently and uniformly (multinomial occupancy law)."""
    dist = np.zeros(len(occs))
    for i, o in enumerate(occs):
        coef = math.factorial(N)
        for c in o:
            coef //= math.factorial(int(c))
        dist[i] = coef * (1.0 / R) ** N
    assert abs(dist.sum() - 1.0) < 1e-9
    return dist


def ladder(od: tuple[str, str], k_extra: int = 8, N: int = 3, K: int = 1) -> dict:
    env = make_multiconvoy_env(od, N=N, K=K, k_extra_routes=k_extra,
                               edge_vuln_band=(0.15, 0.95), absolute_vuln_norm=True,
                               menu_select=True, objective="mission")
    game = env.game
    R = game.n_routes
    occs, M = objective_matrix(game, N)
    sol = solve_multiconvoy(game, N)
    u_stack = exploitability(M, uniform_stack_dist(occs, N))
    u_indep = exploitability(M, uniform_independent_dist(occs, N, R))
    return {"od": od, "R": R, "loss_det": sol.loss_det, "eq": sol.loss_mixed,
            "uniform_stack": u_stack, "uniform_independent": u_indep}


def main() -> None:
    print("=== Missing naive-randomisation rows (mission-failure exploitability, lower better) ===")
    for od in [("35", "159"), ("62", "97")]:
        r = ladder(od)
        print(f"OD {r['od'][0]}->{r['od'][1]} k8 (R={r['R']}, N=3, K=1):")
        print(f"  loss_det (ALNS)        = {r['loss_det']:.3f}")
        print(f"  uniform-INDEPENDENT    = {r['uniform_independent']:.3f}")
        print(f"  uniform-STACK          = {r['uniform_stack']:.3f}   <- the 3-line heuristic")
        print(f"  equilibrium            = {r['eq']:.3f}")

    print("\n=== Menu sufficiency: equilibrium value vs candidate-route-menu size (35->159) ===")
    for k_extra in [0, 4, 8, 12, 16]:
        r = ladder(("35", "159"), k_extra=k_extra)
        print(f"  k_extra={k_extra:>2} (R={r['R']:>2}): eq={r['eq']:.4f}  loss_det={r['loss_det']:.4f}  "
              f"uniform_stack={r['uniform_stack']:.4f}")


if __name__ == "__main__":
    main()
