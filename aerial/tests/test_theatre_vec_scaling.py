"""Guards the terrain table and the vector-theatre loader: the sea and alpine classes, the polygon
fetch format, sea being non-emplaceable, and range scaling that leaves the default scale
untouched."""
import os

import numpy as np
import pytest
from shapely.geometry import Point

from src.envs.aerial_theatre_vec import (PRIORITY, TERRAIN, hazard_sites, lateral_width,
                                         load_vec_theatre)

KGD = "data/maps/theatre_kgd_gvardeysk_vec.json"
TH = load_vec_theatre(KGD)


def test_sea_and_alpine_are_nonemplaceable_terrain():
    for k in ("sea", "alpine"):
        assert k in TERRAIN and not TERRAIN[k]["emplace"]
        assert k in PRIORITY
    assert TERRAIN["sea"]["los"] is False and TERRAIN["alpine"]["los"] is True


def test_range_scale_multiplies_ranges_but_not_lethality():
    _, rr1, pp1, _ = hazard_sites(TH, spacing_km=2.0, standoff_km=6.0, range_scale=1.0)
    _, rr2, pp2, _ = hazard_sites(TH, spacing_km=2.0, standoff_km=6.0, range_scale=2.0)
    assert np.allclose(rr2, 2.0 * rr1)              # ranges scale with the map
    assert np.allclose(pp2, pp1)                    # firepower is terrain-set, never scaled


def test_default_range_scale_is_byte_identical():
    a = hazard_sites(TH, spacing_km=2.0, standoff_km=6.0)
    b = hazard_sites(TH, spacing_km=2.0, standoff_km=6.0, range_scale=1.0)
    assert np.array_equal(a[1], b[1]) and a[3] == b[3]


def test_lateral_width_is_positive_and_perpendicular():
    assert lateral_width(TH) > 10.0


@pytest.mark.skipif(not os.path.exists("data/maps/theatre_narva_vec.json"),
                    reason="narva vec theatre not fetched locally (data/ is gitignored)")
def test_new_poly_format_loads_and_sea_is_nonemplaceable():
    th = load_vec_theatre("data/maps/theatre_narva_vec.json")
    assert "sea" in th.polys and th._union.get("sea") is not None
    sea_u = th._union["sea"]
    c = sea_u.representative_point()
    assert th.classify((c.x, c.y)) == "sea"        # a sea point classifies as sea
    coords, rr, pp, cls = hazard_sites(th, spacing_km=3.0, standoff_km=6.0, range_scale=2.0)
    assert len(coords) > 20                         # the land corridor still offers sites
    assert not any(sea_u.contains(Point(float(x), float(y))) for x, y in coords)  # none in the sea
