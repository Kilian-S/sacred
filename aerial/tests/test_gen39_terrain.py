"""Contracts for the v2 terrain table: v1 defaults, sight-line masking, urban emplacement, the
reveal trigger, and the hide-without-blind asymmetry."""
import numpy as np
from shapely.geometry import LineString

from src.envs.aerial_theatre_vec import (TERRAIN, TERRAIN_V2, blocker_union, build_menu,
                                         containing_blockers, hazard_sites, load_vec_theatre,
                                         reveal_flags, route_survival, terrain_v2)

KGD = "data/maps/theatre_kgd_gvardeysk_vec.json"
TH = load_vec_theatre(KGD)
ROUTES = build_menu(TH, R=5)


def _urban_only_reference(route, coords, rr, pp):
    """Urban-only masking reference for route_survival, the byte-identity oracle for v1."""
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
    """v1 declares forest los=True but masks with urban only; the default keeps the implemented
    behaviour and honouring the declared table is opt-in."""
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


# --- (v) forest HIDES without BLINDING (the v2 default) ---------------------------------------

def test_forest_hides_but_does_not_blind_by_default():
    """Canopy conceals a ground team but does not block its fire; urban does both, being a true
    vertical obstacle."""
    t = terrain_v2()
    assert t["forest"]["reveal"] is False and t["forest"]["los"] is False   # hidden, not blind
    assert t["urban"]["reveal"] is False and t["urban"]["los"] is True      # hidden AND blocking
    assert t["open"]["reveal"] is True and t["field"]["reveal"] is True


def test_default_blocker_excludes_forest_and_the_symmetric_variant_is_still_reachable():
    """Woodland stays out of the sight-line mask by default; forest_los=True restores the
    symmetric rule, which is a different game."""
    default, symmetric = blocker_union(TH, terrain_v2()), blocker_union(TH, terrain_v2(forest_los=True))
    assert symmetric.area > default.area
    assert default.equals(blocker_union(TH, {k: dict(v, los=(k in ("urban", "alpine")))
                                             for k, v in TERRAIN_V2.items()}))


def test_brief_states_hiding_and_sight_blocking_separately():
    """The brief must state hiding and sight-blocking as separate facts, with woodland briefed as
    hiding but not blocking."""
    from src.redforce import _physics_table_text
    txt = _physics_table_text(1.0, terrain_v2())
    assert "conceals you (blocks line of sight)" not in txt
    forest = next(l for l in txt.splitlines() if l.strip().startswith("- forest"))
    assert "stay HIDDEN" in forest and "masks sight lines" not in forest
    urban = next(l for l in txt.splitlines() if l.strip().startswith("- urban"))
    assert "stay HIDDEN" in urban and "masks sight lines" in urban
    open_ = next(l for l in txt.splitlines() if l.strip().startswith("- open"))
    assert "GIVES YOUR POSITION AWAY" in open_


# --- (vi) the concentration must not leak across terrain classes -------------------------------

def test_engagement_concentration_stays_on_the_teams_own_ground():
    """A team's engagement smear must stay within its own terrain class, otherwise a woodland team
    draws reach and lethality from open sites while keeping woodland's invisibility."""
    import collections
    from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
    base = ConcealBase(KGD, terrain=terrain_v2(hidden_leth=0.4, conceal_reach=0.85),
                       range_scale=1.0)
    pp = base.lethality(resample_field(base.coords, 5100), hidden_leth=0.4)
    for c in ("open", "field", "forest", "urban"):
        pool = [i for i, x in enumerate(base.cls) if x == c]
        if not pool:
            continue
        L = np.array(pool[:1])
        for same_class, floor in ((True, 1.0 - 1e-12), (False, 0.0)):
            g = ConcealDyn(base, pp, L, w=2, same_class=same_class)
            share = collections.defaultdict(float)
            for i, x in enumerate(base.cls):
                share[x] += g.prior_j[0][i]
            assert share[c] >= floor, (c, same_class, share[c])
        leaked = ConcealDyn(base, pp, L, w=2, same_class=False)
        own = sum(w for i, w in enumerate(leaked.prior_j[0]) if base.cls[i] == c)
        if c in ("forest", "urban"):
            assert own < 0.35, f"{c} used to leak most of its effect; guard is stale"


def test_same_class_default_is_the_non_leaking_one():
    import inspect

    from src.envs.aerial_conceal import ConcealDyn
    assert inspect.signature(ConcealDyn.__init__).parameters["same_class"].default is True


# --- (vii) force selection scores both pickers -------------------------------------------------

def test_choose_force_scores_both_pickers_and_is_never_worse_than_topk():
    from src.envs.aerial_conceal import ConcealBase, ConcealDyn, choose_force, pick_laydown, \
        resample_field
    base = ConcealBase(KGD, terrain=terrain_v2(), spacing_km=6.0, standoff_km=4.0)
    pp = base.lethality(resample_field(base.coords, 5100))
    L, g, picker = choose_force(base, pp, "open", 2, np.random.default_rng(0), w=2)
    assert picker in ("topk", "comb0", "comb1", "comb2") and len(L) == 2
    L0 = pick_laydown(base, pp, "open", 2, np.random.default_rng(0))
    g0 = ConcealDyn(base, pp, L0, w=2)
    assert g.episodic(T=40) >= g0.episodic(T=40) - 1e-12


def test_spotting_follows_the_fire_and_hidden_teams_never_reveal():
    """A team is spotted within range of any position it fires from, not only its nominal site, so
    the trigger is a superset of the own-site one; concealed ground never reveals."""
    from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
    base = ConcealBase(KGD, terrain=terrain_v2(), spacing_km=6.0, standoff_km=4.0)
    pp = base.lethality(resample_field(base.coords, 5100))
    open_pool = np.where(~base.concealed)[0]
    hid_pool = np.where(base.concealed)[0]
    L = np.concatenate([open_pool[:2], hid_pool[:1]])
    g = ConcealDyn(base, pp, L, w=2)
    old = base.expo[:, L] & base.reveal[L][None, :]
    assert g.revealable[old].all()                      # superset of the own-site trigger
    assert not g.revealable[:, 2].any()                 # the concealed team never reveals
    assert g.revealable[:, :2].sum() >= old[:, :2].sum()


def test_per_team_doctrines_identical_reproduce_the_single_doctrine_game():
    """Uniform per-team doctrines reproduce the single-doctrine game exactly, differing doctrines
    change the aim, and hold_static biases towards track-independence."""
    from src.envs.aerial_conceal import ConcealBase, ConcealDyn, resample_field
    base = ConcealBase(KGD, terrain=terrain_v2(), spacing_km=6.0, standoff_km=4.0)
    pp = base.lethality(resample_field(base.coords, 5100))
    L = np.where(~base.concealed)[0][:3]
    g0 = ConcealDyn(base, pp, L, w=2, tau=0.10, q_rep=0.6, q_flee=0.2, q_ar=0.3)
    same = [dict(q_rep=0.6, q_flee=0.2, q_ar=0.3, tau=0.10, w=2)] * 3
    g1 = ConcealDyn(base, pp, L, w=2, tau=0.10, doctrines=same)
    assert np.allclose(g0.aim, g1.aim, atol=1e-12) and np.allclose(g0.stepdmg, g1.stepdmg,
                                                                   atol=1e-12)
    mixed = [dict(q_rep=1.0, tau=0.05, w=1), dict(q_hold=1.0, tau=0.10),
             dict(q_flee=0.7, q_ar=0.3, tau=0.20, w=2)]
    g2 = ConcealDyn(base, pp, L, w=2, tau=0.10, doctrines=mixed)
    assert not np.allclose(g0.stepdmg, g2.stepdmg, atol=1e-6)
    hold_only = [dict(q_hold=1.0, tau=0.10)] * 3
    g3 = ConcealDyn(base, pp, L, w=2, tau=0.10, doctrines=hold_only)
    spread = np.abs(g3.stepdmg - g3.stepdmg.mean(axis=0, keepdims=True)).max()
    assert spread < np.abs(g0.stepdmg - g0.stepdmg.mean(axis=0, keepdims=True)).max()


def test_force_schema_follows_the_table_in_force():
    from src.redforce import FORCE_SCHEMA, force_schema
    path = lambda s: s["properties"]["agents"]["items"]["properties"]["emplacement_zone"][  # noqa: E731
        "properties"]["terrain"]["enum"]
    assert "urban" not in path(FORCE_SCHEMA)            # v1 does not admit urban emplacement
    assert "urban" in path(force_schema(terrain_v2()))  # v2: urban is choosable
    assert force_schema() is FORCE_SCHEMA               # default untouched
