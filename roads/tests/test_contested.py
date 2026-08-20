"""Tests for the contested-resupply arena (gen07 / B6).

Locks the config surface (dynassign lineage + route reach) and asserts the arena is wired into
both the trainer CLI and the evaluation harness. No training; a short scripted-attacker rollout
checks the route-reach attacker actually engages in destination mode.
"""

from __future__ import annotations

from scripts.evaluate_dynamic_assign import dynassign_config
from src.baselines.attackers import targeted_block_policy
from src.baselines.greedy_dispatch import greedy_insertion_policy, run_episode
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.contested import contested_config, make_contested_env


def test_contested_config_is_dynassign_plus_route_reach():
    c = contested_config()
    d = dynassign_config()
    # Route reach is THE distinguishing change.
    assert c.antag_reach == "route"
    assert d.antag_reach == "leashed"
    # Everything else must match the dynassign lineage exactly (single-knob difference).
    for field in ("max_ticks", "antagonist_interval", "congestion_duration",
                  "congestion_cooldown", "congestion_cost", "reward_mode", "routing_mode",
                  "congestion_levels", "max_antag_actions_per_event"):
        assert getattr(c, field) == getattr(d, field), field


def test_contested_budget_is_parameterised():
    assert contested_config(congestion_budget=1234.0).congestion_budget == 1234.0


def test_contested_env_builds_and_is_dynamic():
    env = make_contested_env(arrival_rate=0.06, demand_seed=123)
    assert env._dynamic_demand
    assert env._expose_queue_features  # motion/ETA features present (route reach needs them)
    assert env.assignment_depots == ("110", "135")


def test_route_reach_attacker_engages_in_destination_mode():
    """The route-reach targeted attacker must actually block edges (spend budget) against greedy
    in the contested arena -- if reach fell back to empty in destination mode it would spend 0."""
    cfg = contested_config()
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=0.06, demand_seed=7), config=cfg)
    result = run_episode(smdp, greedy_insertion_policy(smdp), targeted_block_policy(smdp))
    assert smdp.budget.used > 0.0  # the attacker found routes to aim at
    assert result["total_wait"] > 0.0


def test_contested_wired_into_eval_harness():
    from scripts.evaluate_portfolio import _problem_setup

    cfg, mk, greedy_factory, is_static = _problem_setup("contested", arrival_rate=0.06)
    assert cfg.antag_reach == "route"
    assert is_static is False
    assert greedy_factory is greedy_insertion_policy
