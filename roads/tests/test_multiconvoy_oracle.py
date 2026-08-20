"""Tests for the multi-convoy interdiction oracle (gen08 Phase M)."""
import networkx as nx
import numpy as np

from src.baselines.interdiction_oracle import build_interdiction_game, solve
from src.baselines.multiconvoy_oracle import (
    best_response_attacker_multi, objective_matrix, objective_of, solve_multiconvoy)


def _synthetic_game(K: int = 1):
    """Three edge-disjoint routes s->t of different lengths (hard interception)."""
    G = nx.Graph()
    G.add_edge("s", "a", w=1.0); G.add_edge("a", "t", w=1.0)   # route via a (len 2)
    G.add_edge("s", "b", w=2.0); G.add_edge("b", "t", w=2.0)   # via b (len 4)
    G.add_edge("s", "c", w=3.0); G.add_edge("c", "t", w=3.0)   # via c (len 6)
    return build_interdiction_game(G, "s", "t", K, k_extra=0, weight="w")


def test_n1_mission_reduces_to_single_convoy():
    game = _synthetic_game()
    sc = solve(game)
    mc = solve_multiconvoy(game, N=1, objective="mission")
    assert abs(mc.loss_det - sc.loss_det) < 1e-9
    assert abs(mc.loss_mixed - sc.value) < 1e-6


def test_n1_linear_equals_single_convoy_value():
    game = _synthetic_game()
    sc = solve(game)
    mc = solve_multiconvoy(game, N=1, objective="linear")   # E[frac] with 1 convoy = interception
    assert abs(mc.loss_mixed - sc.value) < 1e-6


def test_reproduces_multiconvoy_probe_110_135():
    from src.envs.multiconvoy_interdiction import make_multiconvoy_env
    game = make_multiconvoy_env(od=("110", "135"), N=2, objective="mission").game
    sol = solve_multiconvoy(game, N=2, objective="mission")
    assert abs(sol.loss_det - 0.728) < 0.02       # scratch/multiconvoy_probe.py, N=2 K=1 soft
    assert abs(sol.loss_mixed - 0.314) < 0.02
    assert sol.gap > 0.3


def test_mission_gap_exceeds_linear_gap():
    """The finding: a loss-averse objective preserves the gap; a risk-neutral one dilutes it."""
    from src.envs.multiconvoy_interdiction import make_multiconvoy_env
    game = make_multiconvoy_env(od=("110", "135"), N=3, objective="mission").game
    mission = solve_multiconvoy(game, N=3, objective="mission")
    linear = solve_multiconvoy(game, N=3, objective="linear")
    assert mission.gap > linear.gap
    assert linear.gap < 0.15        # risk-neutral diluted (deterministic spreading substitutes)
    assert mission.gap > 0.4        # loss-averse strong


def test_best_response_matches_objective_of_and_bounds_value():
    game = _synthetic_game()
    occs, M = objective_matrix(game, N=2, objective="mission")
    d = np.zeros(len(occs)); d[0] = 1.0             # a pure (deterministic) occupancy
    j, loss = best_response_attacker_multi(M, d)
    a = np.zeros(M.shape[1]); a[j] = 1.0
    assert abs(objective_of(M, d, a) - loss) < 1e-9
    sol = solve_multiconvoy(game, N=2, objective="mission")
    assert loss >= sol.loss_mixed - 1e-9            # a pure defender is at least as exploitable
