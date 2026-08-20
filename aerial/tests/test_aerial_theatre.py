"""Real-terrain theatre environment: loading, off-centre endpoints, the forward DAG, a
corridor-spanning menu with no funnel cell, terrain-driven emplacement and line of sight, and a
game that builds and solves non-degenerately."""
import numpy as np
import pytest

from src.baselines.interdiction_oracle import solve
from src.envs.aerial_theatre import (TERRAIN, Theatre, build_route_menu, build_theatre_game,
                                     forward_dag, hazard_sites, load_theatre, route_survival)

TH = load_theatre("data/maps/theatre_kgd_gvardeysk.json")


def test_load_real_terrain_and_offcentre_endpoints():
    assert TH.nrow == 20 and TH.ncol == 46
    assert set(np.unique(TH.grid)).issubset(set(range(5)))
    # endpoints are the real settlements, so they are not vertically centred
    assert TH.base != (TH.nrow // 2, 0) and TH.target != (TH.nrow // 2, TH.ncol - 1)
    assert TH.base[1] < TH.ncol // 2 and TH.target[1] > TH.ncol // 2       # W -> E
    assert TH.base[0] != TH.target[0]                                       # different rows


def test_forward_dag_is_acyclic_toward_target():
    succ, proj = forward_dag(TH)
    for node, nbs in succ.items():
        for m in nbs:
            assert proj[m] > proj[node]                # strictly forward on the base->target axis


def test_menu_spans_corridor_and_reaches_target():
    menu = build_route_menu(TH, R=24, seed=0)
    assert len(menu) >= 18
    assert all(r[0] == TH.base and r[-1] == TH.target for r in menu)
    b = TH.xy(TH.base); t = TH.xy(TH.target); u = (t - b) / np.linalg.norm(t - b)
    nrm = np.array([-u[1], u[0]])
    lat = [float((TH.xy(r[len(r)//2]) - b) @ nrm) / TH.cell_m for r in menu]
    assert max(lat) - min(lat) > 10.0                  # lanes fan across the width (no collapse)
    from collections import Counter
    cnt = Counter(c for r in menu for c in r)
    funnel = [c for c, n in cnt.items() if n >= 0.8 * len(menu) and c not in (TH.base, TH.target)]
    assert not funnel                                  # no single mid cell on all routes


def test_terrain_drives_emplacement_and_standoff():
    coords, rr, pp, cells = hazard_sites(TH, stride=3, standoff_km=7.0)
    for (r, c) in cells:
        assert TERRAIN[TH.cls((r, c))]["emplace"]      # never water/urban
    b, t = TH.xy(TH.base), TH.xy(TH.target)
    for xy in coords:
        assert np.linalg.norm(xy - b) >= 7000 - 1 and np.linalg.norm(xy - t) >= 7000 - 1


def test_los_masking_reduces_exposure():
    # a hazard behind urban cover engages a route less than the same hazard in the open
    menu = build_route_menu(TH, R=24, seed=0)
    coords, rr, pp, _ = hazard_sites(TH, stride=3, standoff_km=4.0)
    S_los = route_survival(TH, menu[0], coords, rr, pp, los=True)
    S_no = route_survival(TH, menu[0], coords, rr, pp, los=False)
    assert np.all(S_los >= S_no - 1e-9)                # masking can only help the defender
    assert S_los.sum() > S_no.sum()                    # and it does, somewhere


def test_game_builds_and_solves_nondegenerate():
    game, menu, coords, rr, pp, S = build_theatre_game(TH, K=1, menu_size=24, site_stride=3,
                                                       standoff_km=4.0)
    assert game.payoff.shape == (len(menu), len(coords))
    sol = solve(game)
    assert 0.05 < sol.value < 0.9                      # non-degenerate single-drone game
    assert sol.loss_det > sol.value + 0.05             # determinism is exploitable
