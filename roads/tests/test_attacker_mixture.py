"""Tests for the scripted-attacker population (mixture)."""

from __future__ import annotations

import pytest

from src.baselines.attackers import ScriptedAttackerMixture, build_scripted_attacker
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.contested import contested_config, make_contested_env


def _members():
    # Weight-only stand-ins (the policy callables are irrelevant to sampling logic).
    return [("a", lambda e: None, 1.0), ("b", lambda e: None, 1.0), ("c", lambda e: None, 2.0)]


def test_sampling_is_deterministic_by_seed():
    m1 = ScriptedAttackerMixture(_members(), seed=3)
    m2 = ScriptedAttackerMixture(_members(), seed=3)
    assert [m1.sample()[0] for _ in range(40)] == [m2.sample()[0] for _ in range(40)]


def test_all_members_are_drawn_and_counts_tracked():
    m = ScriptedAttackerMixture(_members(), seed=1)
    for _ in range(300):
        m.sample()
    assert set(m.counts) == {"a", "b", "c"}
    assert all(v > 0 for v in m.counts.values())
    # The double-weighted member is drawn ~2x the others (loose bound).
    assert m.counts["c"] > m.counts["a"]


def test_rejects_bad_weights():
    with pytest.raises(ValueError):
        ScriptedAttackerMixture([("a", lambda e: None, 0.0)], seed=0)  # sum <= 0
    with pytest.raises(ValueError):
        ScriptedAttackerMixture([], seed=0)  # empty


def test_build_scripted_attacker_names_bind_to_wrapper():
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=0.06, demand_seed=1),
        config=contested_config())
    smdp.reset_decision_env()
    for name in ("targeted", "pathrand", "gateway", "random"):
        pol = build_scripted_attacker(name, smdp, seed=0)
        assert callable(pol)
    with pytest.raises(ValueError):
        build_scripted_attacker("nonesuch", smdp)
