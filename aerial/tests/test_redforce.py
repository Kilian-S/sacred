"""gen33_llm_adversary: the LLM red-force I/O contract (schema, serialiser, dry force, resolver).
Validates the generation half end-to-end without a live model."""
import numpy as np

from src.envs.aerial_theatre_vec import build_theatre_game, lateral_width, load_vec_theatre
from src.redforce import (ARCHETYPES, FORCE_SCHEMA, dry_force, resolve_force_to_sites,
                          serialise_theatre)

TH = load_vec_theatre("data/maps/theatre_kgd_gvardeysk_vec.json")


def _validate(force):
    assert "agents" in force and isinstance(force["agents"], list)
    for a in force["agents"]:
        assert a["archetype"] in ARCHETYPES
        for k in ("terrain", "region"):
            assert k in a["emplacement_zone"]
        for k in ("punish_pattern", "anticipate_flight", "hold_static"):
            assert k in a["doctrine"]
        assert "rationale" in a


def test_schema_shape():
    props = FORCE_SCHEMA["properties"]["agents"]["items"]["properties"]
    assert set(("archetype", "emplacement_zone", "doctrine", "rationale")).issubset(props)


def test_serialiser_shows_physics_and_terrain():
    system, user = serialise_theatre(TH, phase="single", K=1, range_scale=1.0)
    assert "RED force" in system
    assert "km" in user and "punish_pattern" in user and "mission FAILS" in user
    # the physics table names an emplaceable terrain with a range
    assert "field" in user and "forest" in user


def test_dry_force_is_schema_valid_single_and_coordinated():
    _validate(dry_force(K=1, seed=0))
    fc = dry_force(K=3, seed=1, coordinated=True)
    _validate(fc)
    assert len(fc["agents"]) == 3
    for a in fc["agents"]:
        assert "team_role" in a and "team_id" in a


def test_resolver_maps_to_sites_and_normalised_doctrine():
    game, menu, coords, rr, pp, S, lane_idx = build_theatre_game(TH, K=1, n_lanes=14, n_terrain=12)
    cls = [TH.classify(c) for c in coords]
    site_exposure = (1.0 - S).mean(axis=0)
    force = dry_force(K=3, seed=2, coordinated=True)
    res = resolve_force_to_sites(force, TH, coords, cls, site_exposure)
    assert len(res["sites"]) == 3
    for s in res["sites"]:
        assert 0 <= s < len(coords)                 # every site is a real candidate
    for (q_rep, q_flee, q_eq, tau, w) in res["doctrine"]:
        assert abs((q_rep + q_flee + q_eq) - 1.0) < 1e-6   # doctrine is a normalised simplex
        assert tau in (0.05, 0.10, 0.20) and 1 <= w <= 3
