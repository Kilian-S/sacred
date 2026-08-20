"""gen28 v3-theatre CONTINUOUS (vector) env: real polygons load, off-centre continuous
endpoints, in-bounds smooth lanes, terrain emplacement/standoff/LOS, non-degenerate solve."""
import numpy as np
import pytest
from src.baselines.interdiction_oracle import solve
from src.envs.aerial_theatre_vec import (TERRAIN, build_menu, build_theatre_game, hazard_sites,
                                         lane, load_vec_theatre, route_survival)

TH = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")


def test_loads_real_polygons_and_continuous_offcentre_endpoints():
    assert TH.W > 40 and TH.H > 15
    assert sum(len(v) for v in TH.polys.values()) > 400        # hundreds of real polygons
    assert TH.base[0] < TH.W/2 and TH.target[0] > TH.W/2        # W -> E
    assert abs(TH.base[1] - TH.target[1]) > 3.0                 # off-centre, different latitudes
    assert not np.allclose(TH.base[1], TH.H/2)                  # base NOT vertically centred


def test_lanes_are_smooth_in_bounds_and_reach_endpoints():
    menu = build_menu(TH, R=24)
    for r in menu:
        assert np.allclose(r[0], TH.base) and np.allclose(r[-1], TH.target)
        assert r[:, 0].min() >= -0.01 and r[:, 0].max() <= TH.W + 0.01
        assert r[:, 1].min() >= -0.01 and r[:, 1].max() <= TH.H + 0.01
        # smoothness: no hard corners (max turn per step small)
        d = np.diff(r, axis=0); ang = np.arctan2(d[:, 1], d[:, 0])
        assert np.abs(np.diff(ang)).max() < 0.6
    mids = [np.array(r[len(r)//2]) for r in menu]
    u = (TH.target - TH.base); u = u/np.linalg.norm(u); nrm = np.array([-u[1], u[0]])
    lat = [float((m - TH.base) @ nrm) for m in mids]
    assert max(lat) - min(lat) > 8.0                            # lanes fan across the corridor


def test_terrain_emplacement_standoff_and_classify():
    coords, rr, pp, cls = hazard_sites(TH, spacing_km=2.0, standoff_km=6.0)
    assert len(coords) > 30
    for k in cls:
        assert TERRAIN[k]["emplace"] and k in ("open", "field", "forest")
    for xy in coords:
        assert np.linalg.norm(xy - TH.base) >= 6.0 - 1e-6
        assert np.linalg.norm(xy - TH.target) >= 6.0 - 1e-6


def test_los_masking_only_helps_defender():
    menu = build_menu(TH, R=24)
    coords, rr, pp, _ = hazard_sites(TH, spacing_km=2.0, standoff_km=4.0)
    Sl = route_survival(TH, menu[len(menu)//2], coords, rr, pp, los=True)
    Sn = route_survival(TH, menu[len(menu)//2], coords, rr, pp, los=False)
    assert np.all(Sl >= Sn - 1e-9)


def test_game_nondegenerate_on_real_terrain():
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(TH, K=1, n_lanes=14,
                                                                 n_terrain=12, standoff_km=4.0)
    assert len(menu) > len(lane_idx) >= 10                      # lanes + terrain-aware routes
    sol = solve(game)
    assert 0.05 < sol.value < 0.9
    assert sol.loss_det > sol.value + 0.1                       # determinism exploitable


def test_engagement_footprint_shadows():
    from src.envs.aerial_theatre_vec import engagement_footprint
    coords, rr, pp, cls = hazard_sites(TH, spacing_km=2.0, standoff_km=4.0)
    # a site near the city should have a shadowed (non-circular) footprint; a far-rural one round
    areas = []
    for h in range(len(coords)):
        fp = np.array(engagement_footprint(TH, coords[h], rr[h], n_rays=64))
        a = 0.5 * abs(np.dot(fp[:, 0], np.roll(fp[:, 1], 1)) - np.dot(fp[:, 1], np.roll(fp[:, 0], 1)))
        areas.append(a / (np.pi * rr[h] ** 2))                  # fraction of the full disc
    areas = np.array(areas)
    assert areas.min() < 0.9                                    # at least one shadowed footprint
    assert areas.max() <= 1.01                                  # never exceeds the range disc
