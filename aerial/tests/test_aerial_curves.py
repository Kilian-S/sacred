"""Curve machinery and line-integral exposure: the p_max calibration, the bank limit, obstacle
rejection, menu determinism, payoff against brute force, the disjoint-lane closed form, and
greedy best-response validity."""

import numpy as np
import pytest

from src.baselines.aerial_lanes import lane_stack_distributions
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_curves import (build_curve_menu, build_curved_game, curve_survival_matrix,
                                    dense_hazard_grid, lane_curve, lane_offsets, make_curve)
from src.envs.aerial_sector import SectorLattice, greedy_br_hazards

LAT = SectorLattice(ny=9, nx=13)
PINCH = SectorLattice(ny=9, nx=13, blocked=frozenset(
    {(6, j) for j in range(9) if j not in (3, 4, 5)}))


def test_pmax_calibration_straight_transit():
    straight = make_curve(LAT, [4.0] * 5)
    for pmax in (0.3, 0.6, 0.9):
        S = curve_survival_matrix([straight], np.array([[6.0, 4.0]]), r=1.5, p_max=pmax)
        assert 1.0 - S[0, 0] == pytest.approx(pmax, rel=0.01)     # kappa is calibrated to p_max
    S = curve_survival_matrix([straight], np.array([[6.0, 5.2]]), r=1.5, p_max=0.9)
    assert 0.0 < 1.0 - S[0, 0] < 0.9                              # grazing < dead centre
    S = curve_survival_matrix([straight], np.array([[6.0, 5.6]]), r=1.5, p_max=0.9)
    assert 1.0 - S[0, 0] < 1e-6                                   # outside radius: safe


def test_curvature_bound_and_lanes_flyable():
    assert make_curve(LAT, [0.0, 8.0, 0.0, 8.0, 0.0]) is None     # hairpin: not flyable
    for off in lane_offsets(LAT, 1.6) + lane_offsets(LAT, 0.8):
        assert lane_curve(LAT, off) is not None                    # every lane banks legally


def test_obstacles_reject_wall_crossers():
    assert lane_curve(PINCH, 0.0) is None                          # into the wall
    assert lane_curve(PINCH, 4.0) is not None                      # through the gap
    menu, lane_idx = build_curve_menu(PINCH, r=1.6, R=30, seed=0)
    assert len(lane_idx) == 1                                      # only one lane survives a pinch
    rects_hit = [c for c in menu if np.any(
        (np.abs(c.pts[:, 0] - 6.0) < 0.5) & ((c.pts[:, 1] < 2.5) | (c.pts[:, 1] > 5.5)))]
    assert not rects_hit                                           # nobody crosses the wall


def test_menu_deterministic_lanes_first():
    m1, l1 = build_curve_menu(LAT, r=1.2, R=40, seed=0)
    m2, l2 = build_curve_menu(LAT, r=1.2, R=40, seed=0)
    assert [c.offsets for c in m1] == [c.offsets for c in m2] and l1 == l2
    assert len(m1) == 40 and l1 == list(range(len(lane_offsets(LAT, 1.2))))
    assert len({c.offsets for c in m1}) == 40


def test_payoff_matches_bruteforce_integral():
    menu, _ = build_curve_menu(LAT, r=1.5, R=8, seed=0)
    centres = np.array([[4.0, 2.0], [8.0, 6.0], [6.0, 4.0]])
    game, S = build_curved_game(LAT, menu, centres, K=2, r=1.5, p_max=0.8)
    kappa = -np.log(1.0 - 0.8) / 1.5
    for i in (0, 3):
        c = menu[i]
        mids = (c.pts[:-1] + c.pts[1:]) / 2.0
        ds = np.linalg.norm(np.diff(c.pts, axis=0), axis=1)
        for j, iset in enumerate(game.interdiction_sets[:3]):
            lam = 0.0
            for h in iset:
                d = np.linalg.norm(mids - centres[h], axis=1)
                lam += (kappa * np.clip(1.0 - d / 1.5, 0.0, None) * ds).sum()
            assert game.payoff[i, j] == pytest.approx(1.0 - np.exp(-lam), abs=1e-10)


def test_disjoint_lanes_closed_form():
    """Three far-apart lanes with one hazard dead-centre on each give a uniform equilibrium of
    value p_max/3."""
    menu = [lane_curve(LAT, o) for o in (0.0, 4.0, 8.0)]
    centres = np.array([[6.0, 0.0], [6.0, 4.0], [6.0, 8.0]])
    game, S = build_curved_game(LAT, menu, centres, K=1, r=1.0, p_max=0.9)
    sol = solve(game)
    assert sol.value == pytest.approx(0.9 / 3.0, rel=0.03)
    assert np.allclose(sol.defender_strategy, 1 / 3, atol=0.02)
    assert sol.loss_det == pytest.approx(0.9, rel=0.03)


def test_greedy_br_on_integral_exposure():
    menu, _ = build_curve_menu(LAT, r=1.2, R=12, seed=0)
    centres = dense_hazard_grid(LAT, step=1.0)
    game, S = build_curved_game(LAT, menu, centres, K=2, r=1.2, p_max=0.9)
    rng = np.random.default_rng(1)
    for _ in range(4):
        d = rng.random(len(menu)); d /= d.sum()
        _, exact2 = best_response_attacker(game, d)
        _, greedy2 = greedy_br_hazards(S, d, K=2)
        assert greedy2 <= exact2 + 1e-12
        assert greedy2 >= 0.95 * exact2                            # empirical fidelity floor
    g1, _ = build_curved_game(LAT, menu, centres, K=1, r=1.2, p_max=0.9)
    d = rng.random(len(menu)); d /= d.sum()
    assert greedy_br_hazards(S, d, K=1)[1] == pytest.approx(
        best_response_attacker(g1, d)[1], abs=1e-12)               # K=1 exact


def test_lane_stacks_on_curved_game():
    menu, lane_idx = build_curve_menu(LAT, r=0.8, R=30, seed=0)
    centres = dense_hazard_grid(LAT, step=0.5)
    game, S = build_curved_game(LAT, menu, centres, K=1, r=0.8, p_max=0.9)
    dists = lane_stack_distributions(game, lane_idx, S)
    assert set(dists) == {"uniform_lane", "invrisk_lane", "uniform_full", "invrisk_full"}
    for d in dists.values():
        assert d.sum() == pytest.approx(1.0)
    sol = solve(game)
    _, v = best_response_attacker(game, dists["uniform_lane"])
    assert v >= sol.value - 1e-9                                   # nothing beats equilibrium
