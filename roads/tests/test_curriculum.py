"""Tests for the attack exposure and strength curriculum (pure logic, no training)."""

from __future__ import annotations

from src.agents.curriculum import AttackCurriculum


def _curr(**kw):
    base = dict(budget_min=500.0, budget_max=4000.0, n_levels=4, p_attack=1.0,
                competence_floor=0.4, window=5, seed=0)
    base.update(kw)
    return AttackCurriculum(**base)


def test_budget_levels_span_min_to_max():
    c = _curr()
    assert c.current_budget == 500.0  # level 0
    assert c._budget_at(c.n_levels - 1) == 4000.0  # top level
    # Monotone increasing across levels.
    budgets = [c._budget_at(i) for i in range(c.n_levels)]
    assert budgets == sorted(budgets) and len(set(budgets)) == c.n_levels


def test_ramps_up_only_when_competent():
    c = _curr(window=5, competence_floor=0.4)
    # A full window at/above the floor advances exactly one level and resets the window.
    for _ in range(5):
        c.decide()
        c.record(0.6)
    assert c.level == 1
    # A full window BELOW the floor holds the level.
    for _ in range(5):
        c.decide()
        c.record(0.2)
    assert c.level == 1


def test_does_not_exceed_top_level():
    c = _curr(n_levels=2, window=2, competence_floor=0.0)
    for _ in range(100):
        c.decide()
        c.record(1.0)
    assert c.level == 1  # capped at n_levels-1
    assert c.current_budget == 4000.0


def test_clean_episodes_do_not_drive_the_ramp():
    # p_attack=0 -> every episode clean -> competence never recorded -> level frozen at 0.
    c = _curr(p_attack=0.0, window=3, competence_floor=0.0)
    for _ in range(50):
        attacked, budget = c.decide()
        assert attacked is False and budget == 0.0
        c.record(1.0)
    assert c.level == 0


def test_mixing_is_deterministic_by_seed():
    a = _curr(p_attack=0.5, seed=7)
    b = _curr(p_attack=0.5, seed=7)
    seq_a = [a.decide()[0] for _ in range(30)]
    seq_b = [b.decide()[0] for _ in range(30)]
    assert seq_a == seq_b  # reproducible schedule
    assert any(seq_a) and not all(seq_a)  # genuinely mixes clean and attacked


def test_attacked_episode_reports_current_budget():
    c = _curr(p_attack=1.0)
    attacked, budget = c.decide()
    assert attacked is True and budget == c.current_budget
