"""The matrix-free greedy best-response env mode must agree with the exact machinery wherever the
exact path is tractable, and must leave the exact path untouched when the flag is off."""
import numpy as np
import pytest

from src.envs.multiconvoy_interdiction import make_multiconvoy_env

OD, BAND = ("110", "135"), (0.15, 0.95)


def _envs(K, k_extra=0, N=2):
    exact = make_multiconvoy_env(OD, N=N, K=K, k_extra_routes=k_extra, edge_vuln_band=BAND,
                                 absolute_vuln_norm=True, menu_select=True, objective="mission")
    greedy = make_multiconvoy_env(OD, N=N, K=K, k_extra_routes=k_extra, edge_vuln_band=BAND,
                                  absolute_vuln_norm=True, menu_select=True, objective="mission",
                                  greedy_br=True)
    return exact, greedy


def test_flag_off_is_exact_path():
    env = make_multiconvoy_env(OD, N=2, K=1, edge_vuln_band=BAND, menu_select=True)
    assert env.obj_matrix is not None
    assert env.vuln_by_edge == {}


def test_greedy_env_same_routes_and_occupancies():
    exact, greedy = _envs(K=2)
    assert greedy.game.routes == exact.game.routes
    assert len(greedy.occupancies) == len(exact.occupancies)
    assert greedy.obj_matrix is None


def test_route_interception_matches_exact_payoff_columns():
    """route_interception(edge set) must equal the exact K=2 game's payoff column for that iset."""
    exact, greedy = _envs(K=2)
    for j, iset in enumerate(exact.game.interdiction_sets[:25]):
        p_exact = exact.game.payoff[:, j]
        p_greedy = greedy.route_interception(iset)
        assert np.allclose(p_exact, p_greedy, atol=1e-9), (j, p_exact, p_greedy)


def test_exploitability_agrees_at_K1():
    """Greedy best response equals the exact best response at K=1, through the env yardstick."""
    exact, greedy = _envs(K=1)
    rng = np.random.default_rng(0)
    for _ in range(3):
        d = rng.random(len(exact.occupancies)); d /= d.sum()
        assert abs(exact.exploitability_of_occupancy_dist(d)
                   - greedy.exploitability_of_occupancy_dist(d)) < 1e-9


def test_exploitability_bound_at_K2():
    """greedy within [(1 - 1/e) * exact, exact + eps] at K=2 (submodularity guarantee)."""
    exact, greedy = _envs(K=2)
    rng = np.random.default_rng(1)
    for _ in range(3):
        d = rng.random(len(exact.occupancies)); d /= d.sum()
        v_ex = exact.exploitability_of_occupancy_dist(d)
        v_gr = greedy.exploitability_of_occupancy_dist(d)
        assert v_gr <= v_ex + 1e-9
        assert v_gr >= (1 - 1 / np.e) * v_ex - 1e-9


def test_resolve_accepts_committed_edge_set():
    _, greedy = _envs(K=2)
    greedy.reset()
    edges = tuple(list(greedy.vuln_by_edge.keys())[:2])
    greedy.commit_set(edges)
    for _ in range(greedy.config.N):
        greedy.route_convoy_by_index(0)
    out = greedy.resolve()
    assert out.iset_index == -1
    assert 0.0 <= out.objective_value <= 1.0
