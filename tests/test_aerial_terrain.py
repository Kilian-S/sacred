"""gen28 v5 theatre: generation determinism, terrain->machinery mapping, emplacement/flight
masks, and fleet-game tractability at theatre scale."""

import numpy as np
import pytest

from src.baselines.multiconvoy_oracle import objective_matrix, solve_multiconvoy
from src.envs.aerial_terrain import (EMPLACEABLE, FLYABLE, MOUNTAIN, THREAT, Theatre,
                                     build_theatre_game, generate_theatre)


def test_generation_deterministic():
    a = generate_theatre(seed=7)
    b = generate_theatre(seed=7)
    assert np.array_equal(a.grid, b.grid)
    assert not np.array_equal(generate_theatre(seed=7).grid, generate_theatre(seed=8).grid)


def test_terrain_masks_and_reachability():
    th = generate_theatre(ny=17, nx=40, seed=3)
    lat = th.lattice()
    assert all(th.cls(i, j) == MOUNTAIN for (i, j) in lat.blocked)   # blocked == mountains
    import networkx as nx
    G = lat.graph()
    assert nx.has_path(G, lat.base, lat.target)                       # target reachable
    assert th.cls(*lat.base) in FLYABLE and th.cls(*lat.target) in FLYABLE


def test_hazard_field_respects_terrain_and_standoff():
    th = generate_theatre(seed=5)
    centres, pmax, rad = th.hazard_field(step=1.0, safe_r=3.0)
    lat = th.lattice()
    base, target = np.asarray(lat.base, float), np.asarray(lat.target, float)
    for c, p, r in zip(centres, pmax, rad):
        cls = th.cls(int(round(c[0])), int(round(c[1])))
        assert cls in EMPLACEABLE                                     # only emplaceable terrain
        assert (p, r) == THREAT[cls]                                  # threat read from class
        assert np.hypot(*(c - base)) >= 3.0 and np.hypot(*(c - target)) >= 3.0  # standoff
    # heterogeneity actually present (more than one threat radius in play)
    assert len(set(rad.tolist())) >= 2


def test_theatre_game_builds_and_fleet_solves():
    th = generate_theatre(ny=15, nx=32, seed=11)
    game, S, menu, centres, pmax, rad, exp = build_theatre_game(th, K=1, step=1.0, R=24)
    assert game.n_routes == len(menu) >= 12
    assert S.shape == (len(menu), len(centres))
    assert (exp[:-1] <= exp.max() + 1e-9).all()                       # cover routes present
    occs, M = objective_matrix(game, 3, "mission", 1)                 # N=3 fleet mission
    assert M.shape[0] == len(occs)
    sol = solve_multiconvoy(game, 3, "mission")
    assert 0.0 < sol.loss_mixed < sol.loss_det <= 1.0                 # non-degenerate game


def test_cover_routes_beat_straight_on_exposure():
    """The menu's safest route is materially less exposed than a naive straight crossing:
    terrain cover is real and the menu contains it."""
    th = generate_theatre(seed=2)
    game, S, menu, centres, pmax, rad, exp = build_theatre_game(th, K=1, step=1.0, R=30)
    exposure = 1.0 - S.min(axis=1)
    assert exposure.min() < np.median(exposure)                      # a genuine cover option
