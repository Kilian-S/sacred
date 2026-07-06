"""Tests for the interdiction security-game oracle (gen08 ground truth)."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from src.baselines.interdiction_oracle import (
    best_response_attacker, build_interdiction_game, edges_of_route,
    interception_of_distribution, solve)


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
