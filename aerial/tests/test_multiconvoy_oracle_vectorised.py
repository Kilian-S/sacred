"""Regression test: the vectorised objective_matrix closed forms (mission/linear) equal the
generic per-entry loop implementation (objective_value with the Poisson-binomial pmf)."""
from __future__ import annotations

import numpy as np

from src.baselines.interdiction_oracle import InterdictionGame
from src.baselines.multiconvoy_oracle import objective_matrix, objective_value


def _toy_game(R=4, E=6, seed=0):
    rng = np.random.default_rng(seed)
    payoff = rng.uniform(0.05, 0.95, size=(R, E))
    payoff[0, 0] = 1.0  # exercise the p == 1 clamp
    payoff[2, 3] = 0.0
    return InterdictionGame(routes=tuple(tuple(str(i) for i in range(3)) for _ in range(R)),
                            route_edges=tuple(frozenset() for _ in range(R)),
                            interdiction_sets=tuple((frozenset(),) for _ in range(E)),
                            payoff=payoff, travel_cost=np.ones(R), K=1)


def _loop_matrix(game, N, objective, m=1):
    from src.baselines.multiconvoy_oracle import occupancies
    occs = occupancies(game.n_routes, N)
    M = np.zeros((len(occs), game.payoff.shape[1]))
    for oi, occ in enumerate(occs):
        for j in range(game.payoff.shape[1]):
            M[oi, j] = objective_value(occ, game.payoff[:, j], N, objective, m)
    return M


def test_mission_closed_form_equals_loop():
    game = _toy_game()
    for N in (1, 2, 3):
        _, M_vec = objective_matrix(game, N, "mission")
        M_loop = _loop_matrix(game, N, "mission")
        assert np.allclose(M_vec, M_loop, atol=1e-12)


def test_linear_closed_form_equals_loop():
    game = _toy_game()
    _, M_vec = objective_matrix(game, 3, "linear")
    M_loop = _loop_matrix(game, 3, "linear")
    assert np.allclose(M_vec, M_loop, atol=1e-12)


def test_threshold_m2_still_exact():
    game = _toy_game()
    _, M_vec = objective_matrix(game, 3, "threshold", m=2)
    M_loop = _loop_matrix(game, 3, "threshold", m=2)
    assert np.allclose(M_vec, M_loop, atol=1e-12)
