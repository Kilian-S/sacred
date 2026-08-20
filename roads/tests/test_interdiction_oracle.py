"""Tests for the interdiction security-game oracle."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.baselines.interdiction_oracle import (
    best_response_attacker, build_interdiction_game, edges_of_route,
    interception_of_distribution, length_band_vulnerability, solve,
    survival_intercept_fn)


def _synthetic():
    # base S -> FOB T; 3 edge-disjoint routes + 1 sharing S-A with R1.
    G = nx.Graph()
    for p in (["S", "A", "T"], ["S", "B", "T"], ["S", "C", "T"], ["S", "A", "D", "T"]):
        for u, v in zip(p, p[1:]):
            G.add_edge(u, v, w=1.0)
    return G


def test_deterministic_defender_is_fully_exploitable():
    game = build_interdiction_game(_synthetic(), "S", "T", K=1)
    sol = solve(game)
    # One interdictor can always sit on the single committed route -> loss_det = 1.0.
    assert sol.loss_det == pytest.approx(1.0)


def test_mixed_equilibrium_matches_known_value():
    # 3 edge-disjoint routes, K=1 -> the minimax value is 1/3 (attacker catches one route in three).
    game = build_interdiction_game(_synthetic(), "S", "T", K=1)
    sol = solve(game)
    assert sol.value == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert sol.gap == pytest.approx(2.0 / 3.0, abs=1e-6)
    # the defender equilibrium is a genuine mixture (not a pure route).
    assert (sol.defender_strategy > 1e-6).sum() >= 3
    assert sol.defender_strategy.sum() == pytest.approx(1.0)


def test_value_increases_with_more_interdictors():
    G = _synthetic()
    vals = [solve(build_interdiction_game(G, "S", "T", K=k)).value for k in (1, 2, 3)]
    assert vals[0] < vals[1] < vals[2] or vals == sorted(vals)  # monotone non-decreasing
    assert vals[0] == pytest.approx(1 / 3, abs=1e-6)


def test_best_response_attacker_exploits_determinism():
    game = build_interdiction_game(_synthetic(), "S", "T", K=1)
    # a DETERMINISTIC defender (all mass on route 0) is intercepted with certainty.
    det = np.zeros(game.n_routes); det[0] = 1.0
    _, expl_det = best_response_attacker(game, det)
    assert expl_det == pytest.approx(1.0)
    # the equilibrium mixed defender is far less exploitable.
    sol = solve(game)
    _, expl_mixed = best_response_attacker(game, sol.defender_strategy)
    assert expl_mixed == pytest.approx(sol.value, abs=1e-6)
    assert expl_mixed < expl_det


def test_interception_of_distribution_bilinear():
    game = build_interdiction_game(_synthetic(), "S", "T", K=1)
    sol = solve(game)
    # equilibrium vs equilibrium = the game value.
    v = interception_of_distribution(game, sol.defender_strategy, sol.attacker_strategy)
    assert v == pytest.approx(sol.value, abs=1e-6)


def test_kaliningrad_high_connectivity_has_large_gap():
    from src.envs.assignment_factory import _DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS
    from src.utils.graph_utils import load_osm_graph_and_demands
    nodes, edges = load_osm_graph_and_demands(_DEFAULT_NODES, _DEFAULT_EDGES, _DEFAULT_TASKS)
    G = nx.Graph()
    for u, v, d in edges:
        G.add_edge(str(u), str(v), w=float(d.get("distance", 1.0)))
    sol = solve(build_interdiction_game(G, "33", "71", K=1))
    assert sol.loss_det == pytest.approx(1.0)      # deterministic route fully exploitable
    assert sol.value <= 0.2 + 1e-9                  # edge-conn 6 -> mixed <= ~1/6
    assert sol.gap >= 0.8 - 1e-9                     # the large positive robustness gap


def test_edges_of_route():
    assert edges_of_route(("a", "b", "c")) == {frozenset({"a", "b"}), frozenset({"b", "c"})}


# --- Heterogeneous edge vulnerability: soft interception, non-uniform equilibria ---


def _heterogeneous():
    # 3 edge-disjoint routes with DISTINCT edge lengths -> distinct vulnerabilities under the band map.
    G = nx.Graph()
    G.add_edge("S", "A", w=4.0); G.add_edge("A", "T", w=2.0)
    G.add_edge("S", "B", w=1.0); G.add_edge("B", "T", w=2.0)
    G.add_edge("S", "C", w=2.0); G.add_edge("C", "T", w=3.0)
    return G


def test_length_band_vulnerability_maps_lengths_into_band():
    G = _heterogeneous()
    edges = [frozenset(e) for e in G.edges()]
    vuln = length_band_vulnerability(G, edges, band=(0.2, 0.9))
    assert set(vuln) == set(edges)
    assert vuln[frozenset({"S", "B"})] == pytest.approx(0.2)   # shortest edge -> band lo
    assert vuln[frozenset({"S", "A"})] == pytest.approx(0.9)   # longest edge -> band hi
    # affine in length: w=2 sits at (2-1)/(4-1) of the band.
    assert vuln[frozenset({"A", "T"})] == pytest.approx(0.2 + 0.7 / 3.0)
    # degenerate all-equal lengths -> band midpoint.
    H = nx.Graph(); H.add_edge("x", "y", w=1.0); H.add_edge("y", "z", w=1.0)
    mid = length_band_vulnerability(H, [frozenset({"x", "y"}), frozenset({"y", "z"})], band=(0.2, 0.9))
    assert all(v == pytest.approx(0.55) for v in mid.values())
    # a DESCENDING band inverts the correlation: shortest edge -> highest vulnerability.
    inv = length_band_vulnerability(G, edges, band=(0.9, 0.2))
    assert inv[frozenset({"S", "B"})] == pytest.approx(0.9)
    assert inv[frozenset({"S", "A"})] == pytest.approx(0.2)


def test_survival_intercept_fn_multi_edge():
    vuln = {frozenset({"S", "A"}): 0.9, frozenset({"A", "T"}): 0.4}
    fn = survival_intercept_fn(vuln)
    route = frozenset({frozenset({"S", "A"}), frozenset({"A", "T"})})
    # K=1 reduces to p_e; a missed route is 0; two hits compose by independent survival.
    assert fn(route, (frozenset({"S", "A"}),)) == pytest.approx(0.9)
    assert fn(route, (frozenset({"X", "Y"}),)) == pytest.approx(0.0)
    both = (frozenset({"S", "A"}), frozenset({"A", "T"}))
    assert fn(route, both) == pytest.approx(1.0 - 0.1 * 0.6)


def test_cost_constrained_frontier():
    # heterogeneous costs, HARD interception: S-A-T costs 6, S-B-T costs 3, S-C-T costs 5.
    from src.baselines.interdiction_oracle import cost_constrained_value
    G = _heterogeneous()
    game = build_interdiction_game(G, "S", "T", K=1, k_extra=0)
    sol = solve(game)
    # unconstrained budget reproduces the equilibrium (1/3 on three disjoint routes).
    v_inf, x_inf = cost_constrained_value(game, budget=100.0)
    assert v_inf == pytest.approx(sol.value, abs=1e-6) == pytest.approx(1 / 3, abs=1e-6)
    assert x_inf.sum() == pytest.approx(1.0)
    # the tightest feasible budget forces the cheapest pure route: fully exploitable.
    v_min, x_min = cost_constrained_value(game, budget=3.0)
    assert v_min == pytest.approx(1.0)
    # the frontier is monotone non-increasing in budget, strictly between the endpoints mid-way.
    v_mid, _ = cost_constrained_value(game, budget=4.0)
    assert 1 / 3 < v_mid < 1.0
    assert v_min >= v_mid >= v_inf
    with pytest.raises(ValueError):
        cost_constrained_value(game, budget=2.0)


def test_soft_equilibrium_matches_closed_form():
    # On disjoint routes with per-route max vulnerability p_i*, the attacker's dominant edge per
    # route is its max-p edge, so the game reduces to: value = 1 / sum_i(1/p_i*), defender
    # d_i ~ 1/p_i* (equalising d_i * p_i*). The LP must reproduce this exactly.
    G = _heterogeneous()
    edges = [frozenset(e) for e in G.edges()]
    vuln = length_band_vulnerability(G, edges, band=(0.2, 0.9))
    game = build_interdiction_game(G, "S", "T", K=1, k_extra=0,
                                   intercept_fn=survival_intercept_fn(vuln))
    sol = solve(game)
    p_star = np.array([max(vuln[e] for e in re) for re in game.route_edges])
    inv = 1.0 / p_star
    assert sol.value == pytest.approx(1.0 / inv.sum(), abs=1e-6)
    np.testing.assert_allclose(sol.defender_strategy, inv / inv.sum(), atol=1e-6)
    # the equilibrium is genuinely non-uniform.
    assert sol.defender_strategy.max() > sol.defender_strategy.min() + 0.1
    # loss_det < 1 under soft interception (the best deterministic route survives sometimes),
    # and the mixed equilibrium still beats it.
    assert sol.loss_det == pytest.approx(p_star.min(), abs=1e-9)
    assert sol.value < sol.loss_det
