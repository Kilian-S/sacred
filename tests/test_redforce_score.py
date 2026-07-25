"""gen33_llm_adversary: the force scorer (pinned semantics) against the gen32 machinery.
The binding check: flat prior + single agent + (q_rep, q_flee, 0) at the screened operating
point reproduces scratch DynTheatre exactly (same field seed, same window maths)."""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from src.redforce_score import (ScoreBase, force_aim, force_value, heuristic_force,
                                random_force, resample_field)

_spec = importlib.util.spec_from_file_location(
    "g32", Path(__file__).resolve().parents[1] / "scratch" / "gen32_theatre_hunt.py")
g32 = importlib.util.module_from_spec(_spec)
sys.modules["g32"] = g32
_spec.loader.exec_module(g32)

BASE = ScoreBase("data/maps/theatre_kgd_gvardeysk_vec.json")


def test_field_identity_with_gen32():
    ours = resample_field(BASE.coords, 5100)
    theirs = g32.resample_field(BASE.coords, 5100)
    assert np.allclose(ours, theirs)


def test_flat_prior_single_agent_reproduces_dyntheatre():
    tb = g32.TheatreBase()
    field = g32.resample_field(tb.coords, 5100)
    ref = g32.DynTheatre(tb, field, 2, 0.10, 0.7, 0.3)
    fc = BASE.field(5100)
    A, ctx = force_aim(fc, [0], [(0.7, 0.3, 0.0, 0.10, 2)], sigma_km=None)
    stepdmg = A @ fc.dmg.T
    assert np.allclose(stepdmg, ref.stepdmg, atol=1e-10)
    v = force_value(fc, [0], [(0.7, 0.3, 0.0, 0.10, 2)], sigma_km=None)
    assert abs(v - ref.history_opt()) < 1e-8


def test_mixture_aim_is_normalised_and_prior_concentrates():
    fc = BASE.field(5100)
    sites, doctrine = heuristic_force(BASE, 3)
    A, ctx = force_aim(fc, sites, doctrine, sigma_km=4.0)
    assert A.shape == (len(ctx.states), BASE.H)
    assert np.allclose(A.sum(axis=1), 1.0)
    # tighter prior concentrates aim mass nearer the agents' sites
    d2 = np.stack([((BASE.coords - BASE.coords[s]) ** 2).sum(axis=1) for s in sites]).min(axis=0)
    A_wide, _ = force_aim(fc, sites, doctrine, sigma_km=None)
    near = d2 < np.median(d2)
    assert A[:, near].sum() > A_wide[:, near].sum()


def test_baseline_constructors_are_schema_shaped():
    rng = np.random.default_rng(0)
    sites, doctrine = random_force(BASE, 3, rng)
    assert len(sites) == 3 and all(0 <= s < BASE.H for s in sites)   # stacking allowed
    for (qr, qf, qe, tau, w) in doctrine:
        assert abs(qr + qf + qe - 1.0) < 1e-9 and tau in (0.05, 0.10, 0.20) and 1 <= w <= 3
    hs, hd = heuristic_force(BASE, 3)
    assert len(hs) == 3 and all(d == (0.7, 0.3, 0.0, 0.10, 2) for d in hd)
