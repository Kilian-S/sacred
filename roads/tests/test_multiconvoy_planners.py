"""Tests for the multi-convoy classical baselines (gen08 Phase M / Obj-5)."""
import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import build_interdiction_game
from src.baselines.multiconvoy_oracle import solve_multiconvoy
from src.baselines.multiconvoy_planners import (
    alns_fleet_planner, classical_baselines, shortest_path_fleet)


def _synthetic():
    G = nx.Graph()
    G.add_edge("s", "a", w=1.0); G.add_edge("a", "t", w=1.0)
    G.add_edge("s", "b", w=2.0); G.add_edge("b", "t", w=2.0)
    G.add_edge("s", "c", w=3.0); G.add_edge("c", "t", w=3.0)
    return build_interdiction_game(G, "s", "t", 1, k_extra=0, weight="w")


def test_alns_reaches_loss_det_synthetic():
    game = _synthetic()
    for N in (2, 3, 4):
        sol = solve_multiconvoy(game, N, "mission")
        plan = alns_fleet_planner(game, N, "mission", seed=0)
        assert abs(plan.exploitability - sol.loss_det) < 1e-6


def test_alns_reaches_loss_det_kaliningrad():
    from src.envs.multiconvoy_interdiction import make_multiconvoy_env
    game = make_multiconvoy_env(od=("110", "135"), N=3, objective="mission").game
    sol = solve_multiconvoy(game, 3, "mission")
    plan = alns_fleet_planner(game, 3, "mission", seed=1)
    assert abs(plan.exploitability - sol.loss_det) < 1e-6


def test_shortest_path_stacks_on_cheapest():
    game = _synthetic()
    sp = shortest_path_fleet(game, 3)
    assert len(sp) == 3 and len(set(sp)) == 1
    assert sp[0] == int(np.argmin(game.travel_cost))


def test_baseline_ordering_sacred_beats_alns():
    from src.envs.multiconvoy_interdiction import make_multiconvoy_env
    game = make_multiconvoy_env(od=("110", "135"), N=3, objective="mission").game
    b = classical_baselines(game, 3, "mission")
    assert abs(b["alns"] - b["optimal_deterministic"]) < 1e-6          # ALNS reaches the optimum
    assert b["shortest_path"] >= b["optimal_deterministic"] - 1e-9     # naive no better than optimal
    assert b["equilibrium"] < b["alns"] - 1e-6                         # SACRED strictly beats the ALNS
