"""gen28 aerial sector: lattice determinism, proximity correctness, payoff fidelity,
closed-form equilibrium sanity, greedy-BR validity (submodularity tested, not assumed),
lane machinery. All oracle-side; no training."""

import itertools

import numpy as np
import pytest

from src.baselines.aerial_lanes import (lane_menu_indices, lane_rows,
                                        lane_stack_distributions, tabular_smooth_fp)
from src.baselines.interdiction_oracle import best_response_attacker, solve
from src.envs.aerial_sector import (SectorLattice, arc_hazard_prob, arc_midpoints,
                                    build_aerial_game, build_aerial_menu, coverage_fraction,
                                    greedy_br_hazards, hazard_grid, lane_path, path_length,
                                    route_survival_matrix, weather_cost_penalty)

LAT = SectorLattice(ny=9, nx=13)


def test_lattice_dag_and_determinism():
    G1, G2 = LAT.graph(), LAT.graph()
    assert list(G1.edges) == list(G2.edges)
    assert all(b[0] == a[0] + 1 for a, b in G1.edges)          # strictly forward in depth
    menu1, menu2 = build_aerial_menu(LAT, R=20), build_aerial_menu(LAT, R=20)
    assert menu1 == menu2 and len(menu1) == len(set(menu1)) == 20


def test_lane_paths():
    for row in range(LAT.ny):
        p = lane_path(LAT, row)
        assert p is not None and p[0] == LAT.base and p[-1] == LAT.target
        assert max(abs(n[1] - row) for n in p) <= abs(LAT.base[1] - row)  # reaches the lane
        assert any(n[1] == row for n in p)
        off = abs(row - LAT.base[1])
        assert path_length(p) == pytest.approx((LAT.nx - 1) + 2 * off * (np.sqrt(2) - 1))
    menu = build_aerial_menu(LAT, R=20)
    assert all(lane_path(LAT, row) in menu for row in range(LAT.ny))      # lanes are menu members


def test_lane_path_blocked_and_depth_limits():
    assert lane_path(SectorLattice(ny=9, nx=5), 0) is None                # excursion cannot fit
    lat = SectorLattice(ny=9, nx=13, blocked=frozenset({(6, 8)}))
    assert lane_path(lat, 8) is None                                      # blocked lane row
    assert lane_path(lat, 0) is not None


def test_proximity_linear_and_gauss():
    mids = np.array([[5.0, 4.0]])
    c = np.array([[5.0, 4.0]])
    assert arc_hazard_prob(mids, c, r=2.0, p_max=0.9)[0, 0] == pytest.approx(0.9)
    c2 = np.array([[5.0, 5.0]])                                           # distance 1 = r/2
    assert arc_hazard_prob(mids, c2, r=2.0, p_max=0.9)[0, 0] == pytest.approx(0.45)
    c3 = np.array([[5.0, 6.5]])                                           # beyond r
    assert arc_hazard_prob(mids, c3, r=2.0, p_max=0.9)[0, 0] == 0.0
    g = arc_hazard_prob(mids, c2, r=2.0, p_max=0.9, taper="gauss")[0, 0]
    assert g == pytest.approx(0.9 * np.exp(-1.0 / 2.0))                   # sigma = r/2 = 1
    assert arc_hazard_prob(mids, c3, r=2.0, p_max=0.9, taper="gauss")[0, 0] == 0.0


def test_payoff_matches_bruteforce_loop():
    menu = build_aerial_menu(LAT, R=8)
    centres = hazard_grid(LAT, cols=(4, 8), rows=(2, 4, 6))
    game = build_aerial_game(LAT, menu, centres, K=2, r=1.5)
    for i, path in enumerate(menu):
        mids = arc_midpoints(path)
        p = arc_hazard_prob(mids, centres, 1.5, 0.9)
        for j, iset in enumerate(game.interdiction_sets[:10]):
            surv = 1.0
            for h in iset:
                for a in range(len(mids)):
                    surv *= 1.0 - p[a, h]
            assert game.payoff[i, j] == pytest.approx(1.0 - surv, abs=1e-12)


def test_disjoint_lanes_uniform_equilibrium():
    """3 pure lanes spaced > 2r, one hazard candidate dead-centre on each: symmetric coverage,
    so the equilibrium is uniform over lanes with value = single-lane exposure / 3."""
    lat = SectorLattice(ny=9, nx=13)
    menu = [lane_path(lat, row) for row in (0, 4, 8)]
    centres = np.array([[6.0, 0.0], [6.0, 4.0], [6.0, 8.0]])
    game = build_aerial_game(lat, menu, centres, K=1, r=1.0)
    S = route_survival_matrix(menu, centres, r=1.0, p_max=0.9)
    exposure = 1.0 - S.min(axis=1)
    assert exposure.std() < 1e-9                                          # symmetric by design
    sol = solve(game)
    assert sol.value == pytest.approx(exposure[0] / 3.0, rel=1e-6)
    assert np.allclose(sol.defender_strategy, 1.0 / 3.0, atol=1e-6)
    assert sol.loss_det == pytest.approx(exposure[0], rel=1e-6)


def test_greedy_br_exact_at_k1_and_bound_at_k2():
    menu = build_aerial_menu(LAT, R=12)
    centres = hazard_grid(LAT, cols=(3, 6, 9), rows=(1, 3, 5, 7))
    S = route_survival_matrix(menu, centres, r=1.5, p_max=0.9)
    rng = np.random.default_rng(0)
    for _ in range(5):
        d = rng.random(len(menu)); d = d / d.sum()
        g1 = build_aerial_game(LAT, menu, centres, K=1, r=1.5)
        _, exact1 = best_response_attacker(g1, d)
        _, greedy1 = greedy_br_hazards(S, d, K=1)
        assert greedy1 == pytest.approx(exact1, abs=1e-12)                # K=1 exact
        g2 = build_aerial_game(LAT, menu, centres, K=2, r=1.5)
        _, exact2 = best_response_attacker(g2, d)
        _, greedy2 = greedy_br_hazards(S, d, K=2)
        assert greedy2 <= exact2 + 1e-12
        assert greedy2 >= (1.0 - 1.0 / np.e) * exact2 - 1e-12             # certified bound
        assert greedy2 >= 0.95 * exact2                                    # measured fidelity


def test_weather_is_cost_only():
    menu = build_aerial_menu(LAT, R=10)
    centres = hazard_grid(LAT, cols=(4, 8), rows=(2, 6))
    cells = [((6.0, 4.0), 2.0, 5.0)]
    g0 = build_aerial_game(LAT, menu, centres, K=1, r=1.5)
    g1 = build_aerial_game(LAT, menu, centres, K=1, r=1.5, weather=cells)
    assert np.array_equal(g0.payoff, g1.payoff)                           # never touches the game
    pen = weather_cost_penalty(menu, cells)
    assert np.allclose(g1.travel_cost, g0.travel_cost + pen)
    mid_lane = menu.index(lane_path(LAT, 4))
    assert pen[mid_lane] > 0.0                                            # centre lane pays
    edge_lane = menu.index(lane_path(LAT, 0))
    assert pen[edge_lane] == pytest.approx(0.0)                           # far lane does not


def test_lane_rows_formula_and_stacks():
    assert lane_rows(LAT, r=2.0) == [0, 4, 8]                             # floor(8/4)+1 = 3
    assert len(lane_rows(LAT, r=0.8)) == 6                                # floor(8/1.6)+1 = 6
    menu = build_aerial_menu(LAT, R=20)
    centres = hazard_grid(LAT)
    S = route_survival_matrix(menu, centres, r=2.0, p_max=0.9)
    game = build_aerial_game(LAT, menu, centres, K=1, r=2.0)
    idx = lane_menu_indices(LAT, menu, r=2.0)
    assert len(idx) == 3
    dists = lane_stack_distributions(game, idx, S)
    for name, d in dists.items():
        assert d.sum() == pytest.approx(1.0), name
        _, v = best_response_attacker(game, d)
        assert 0.0 <= v <= 1.0
    assert set(np.nonzero(dists["uniform_lane"])[0]) == set(idx)


def test_tabular_fp_reaches_equilibrium():
    menu = build_aerial_menu(LAT, R=12)
    centres = hazard_grid(LAT, cols=(4, 8), rows=(0, 2, 4, 6, 8))
    game = build_aerial_game(LAT, menu, centres, K=2, r=1.2)
    sol = solve(game)
    value, avg = tabular_smooth_fp(game, rounds=3000)
    assert value <= sol.value * 1.05 + 1e-9                               # within 5% of minimax
    assert value >= sol.value - 1e-9


def test_banded_pmax_broadcast():
    from src.envs.aerial_sector import banded_pmax
    centres = hazard_grid(LAT, cols=(4,), rows=(0, 4, 8))
    pm = banded_pmax(centres, LAT.ny, band=(0.5, 0.95))
    assert pm == pytest.approx([0.95, 0.725, 0.5])                        # south hot, north cold
    mids = np.array([[4.0, 0.0], [4.0, 4.0], [4.0, 8.0]])
    p = arc_hazard_prob(mids, centres, r=1.0, p_max=pm)
    assert p[0, 0] == pytest.approx(0.95) and p[2, 2] == pytest.approx(0.5)
    S_vec = route_survival_matrix(build_aerial_menu(LAT, R=5), centres, r=1.5, p_max=pm)
    S_hot = route_survival_matrix(build_aerial_menu(LAT, R=5), centres, r=1.5, p_max=0.95)
    assert np.all(S_vec >= S_hot - 1e-12)                                 # weaker hazards, safer


def test_coverage_fraction():
    assert coverage_fraction(2, 2.0, 8.0) == pytest.approx(1.0)
    assert coverage_fraction(1, 0.8, 8.0) == pytest.approx(0.2)
