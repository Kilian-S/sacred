"""gen39_concealment: terrain table v2 contracts.

Guards the four things the act depends on: (i) v1 is untouched and remains the default, so
gen31/gen32/gen33 reproduce byte-identically; (ii) forest actually blocks line of sight under v2
(v1 declared the flag and only the LLM brief ever read it); (iii) urban is emplaceable AND not
self-masked (without the own-polygon exemption every urban site is dead); (iv) the reveal trigger
is EXPOSURE with line of sight, not a kill.
"""
import numpy as np
from shapely.geometry import LineString

from src.envs.aerial_theatre_vec import (TERRAIN, TERRAIN_V2, blocker_union, build_menu,
                                         containing_blockers, hazard_sites, load_vec_theatre,
                                         reveal_flags, route_survival, terrain_v2)

KGD = "data/maps/theatre_kgd_gvardeysk_vec.json"
TH = load_vec_theatre(KGD)
ROUTES = build_menu(TH, R=5)


def _urban_only_reference(route, coords, rr, pp):
    """The pre-gen39 route_survival, copied verbatim: the byte-identity oracle for the default
    path. If gen39 ever changes v1 behaviour, this fires."""
    mids = (route[:-1] + route[1:]) / 2.0
    ds = np.linalg.norm(np.diff(route, axis=0), axis=1)
    kappa = -np.log(np.clip(1.0 - pp, 1e-12, 1.0)) / np.clip(rr, 1e-9, None)
    S = np.ones(len(coords))
    urb = TH._urban_union
    for h in range(len(coords)):
        d = np.linalg.norm(mids - coords[h], axis=1)
        taper = np.clip(1.0 - d / rr[h], 0.0, None)
        if urb is not None and not urb.is_empty:
            for a in np.where(taper > 0)[0]:
                if LineString([tuple(coords[h]), tuple(mids[a])]).intersects(urb):
                    taper[a] = 0.0
        S[h] = np.exp(-(kappa[h] * taper * ds).sum())
    return S


# --- (i) v1 untouched -------------------------------------------------------------------------

def test_v1_table_unchanged_and_still_the_default():
    assert TERRAIN["open"] == dict(emplace=True, r_km=2.5, p_max=0.90, los=False)
    assert TERRAIN["urban"]["emplace"] is False          # v1: no emplacement in towns
    a = hazard_sites(TH, spacing_km=4.0, standoff_km=4.0)
    b = hazard_sites(TH, spacing_km=4.0, standoff_km=4.0, terrain=TERRAIN)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and a[3] == b[3]


def test_the_v1_mismatch_is_documented_not_silently_repaired():
    """v1's table DECLARES forest los=True but route_survival only ever masked with urban. That
    mismatch is the gen39 finding; the default path must keep the IMPLEMENTED behaviour so banked
    numbers are untouched, and honouring the declared table must be opt-in."""
    assert TERRAIN["forest"]["los"] is True                     # declared
    declared = blocker_union(TH, TERRAIN)
    assert declared.area > TH._urban_union.area                 # ... and it would mask more
    coords, rr, pp, _ = hazard_sites(TH, spacing_km=4.0, standoff_km=4.0)
    default = route_survival(TH, ROUTES[0], coords, rr, pp, los=True)          # implemented v1
    honoured = route_survival(TH, ROUTES[0], coords, rr, pp, los=True, terrain=TERRAIN)
    assert not np.array_equal(default, honoured)                # the two really do differ
    assert np.all(honoured >= default - 1e-12)                  # more masking -> more survival


def test_default_path_is_byte_identical_to_pre_gen39_urban_only_masking():
    coords, rr, pp, _ = hazard_sites(TH, spacing_km=4.0, standoff_km=4.0)
    ref = _urban_only_reference(ROUTES[0], coords, rr, pp)
    assert np.array_equal(route_survival(TH, ROUTES[0], coords, rr, pp, los=True), ref)


# --- (ii) forest blocks line of sight under v2 ------------------------------------------------

def test_v2_blocker_is_urban_plus_forest():
    urban, v2 = TH._urban_union, blocker_union(TH, terrain_v2(forest_los=True))
    assert v2.area > urban.area                         # urban + forest > urban
    assert v2.contains(urban.buffer(-1e-9))


def test_forest_los_toggle_is_respected():
    off = blocker_union(TH, terrain_v2(forest_los=False))
    assert abs(off.area - TH._urban_union.area) < 1e-6


# --- (iii) urban emplaceable, and NOT self-masked ---------------------------------------------

def test_v2_admits_urban_sites_and_v1_does_not():
    t2 = terrain_v2()
    _, _, _, cls1 = hazard_sites(TH, spacing_km=2.0, standoff_km=4.0)
    _, _, _, cls2 = hazard_sites(TH, spacing_km=2.0, standoff_km=4.0, terrain=t2)
    assert "urban" not in cls1 and "urban" in cls2
    assert len(cls2) > len(cls1)


def test_urban_sites_can_still_engage_the_own_polygon_exemption():
    """Without the exemption every urban site starts its sightline inside the blocker and is
    dead (survival identically 1 against every route)."""
    t2 = terrain_v2()
    coords, rr, pp, cls = hazard_sites(TH, spacing_km=2.0, standoff_km=4.0, terrain=t2)
    own = containing_blockers(TH, coords, t2)
    urban = [h for h, c in enumerate(cls) if c == "urban"]
    assert urban, "expected urban candidate sites under v2"
    assert all(own[h] is not None for h in urban)       # each urban site stands in a polygon
    alive = np.zeros(len(coords), dtype=bool)
    for r_ in ROUTES:
        S = route_survival(TH, r_, coords, rr, pp, los=True, terrain=t2, own_polys=own)
        alive |= S < 1.0 - 1e-12
    assert alive[urban].any(), "every urban site was self-masked: the exemption is not working"


def test_own_polygon_exemption_only_relaxes_never_tightens():
    t2 = terrain_v2()
    coords, rr, pp, _ = hazard_sites(TH, spacing_km=3.0, standoff_km=4.0, terrain=t2)
    own = containing_blockers(TH, coords, t2)
    with_ex = route_survival(TH, ROUTES[2], coords, rr, pp, los=True, terrain=t2, own_polys=own)
    without = route_survival(TH, ROUTES[2], coords, rr, pp, los=True, terrain=t2, own_polys=None)
    assert np.all(with_ex <= without + 1e-12)           # exemption can only ADD engagements


# --- (iv) the reveal mechanic ------------------------------------------------------------------

def test_reveal_flags_follow_the_terrain_class():
    t2 = terrain_v2()
    _, _, _, cls = hazard_sites(TH, spacing_km=2.0, standoff_km=4.0, terrain=t2)
    rev = reveal_flags(cls, t2)
    for h, c in enumerate(cls):
        assert rev[h] == (c in ("open", "field"))
    assert reveal_flags(cls, TERRAIN).all()             # v1 has no concealment: everything reveals


def test_exposure_flag_is_engagement_with_line_of_sight_not_a_kill():
    t2 = terrain_v2()
    coords, rr, pp, _ = hazard_sites(TH, spacing_km=3.0, standoff_km=4.0, terrain=t2)
    own = containing_blockers(TH, coords, t2)
    S, exposed = route_survival(TH, ROUTES[1], coords, rr, pp, los=True, terrain=t2,
                                own_polys=own, return_exposed=True)
    # a site is 'exposed' exactly when it could put non-zero hazard on the flight
    assert np.array_equal(exposed, S < 1.0 - 1e-12)
    assert exposed.any() and not exposed.all()


def test_hidden_lethality_knob_scales_only_the_concealed_classes():
    t = terrain_v2(hidden_leth=0.5)
    assert t["forest"]["p_max"] == TERRAIN_V2["forest"]["p_max"] * 0.5
    assert t["urban"]["p_max"] == TERRAIN_V2["urban"]["p_max"] * 0.5
    assert t["open"]["p_max"] == TERRAIN_V2["open"]["p_max"]
    assert t["field"]["p_max"] == TERRAIN_V2["field"]["p_max"]
    assert t["open"]["r_km"] > t["field"]["r_km"] > t["forest"]["r_km"] > t["urban"]["r_km"]
