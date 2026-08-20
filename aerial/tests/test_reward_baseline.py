"""Tests the counterfactual twin reward baseline on the contested arena, where subtracting an
action-independent per-tick baseline b(t) shifts the episode return by the per-episode constant
sum_t b(t) and so leaves the zero-sum game and its equilibrium intact.
"""

from __future__ import annotations

import dataclasses

from src.baselines.attackers import targeted_block_policy
from src.baselines.greedy_dispatch import (
    greedy_insertion_policy, no_antagonist_policy, run_episode)
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.contested import (
    contested_config, make_contested_env, make_greedy_twin_baseline_provider)

SEED = 424242
RATE = 0.06


def _contested_smdp(reward_baseline="none", demand_seed=SEED):
    cfg = contested_config()
    provider = None
    if reward_baseline == "twin":
        cfg = dataclasses.replace(cfg, reward_baseline="twin")
        provider = make_greedy_twin_baseline_provider(cfg, arrival_rate=RATE)
    return SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=RATE, demand_seed=demand_seed),
        config=cfg, baseline_provider=provider)


def _greedy_run(smdp, attacker_of):
    # the attacker must be bound to the wrapper the episode drives, since it reads smdp.env on
    # every event and a separate wrapper instance would desynchronise it
    return run_episode(smdp, greedy_insertion_policy(smdp), attacker_of(smdp))


def test_twin_baseline_shifts_return_by_the_twin_constant():
    # greedy against a targeted attacker, no baseline
    real_wait = _greedy_run(_contested_smdp("none"), targeted_block_policy)["total_wait"]

    # the same episode with the twin baseline. Greedy is deterministic and the baseline changes
    # only the reward, never the dynamics or the actions, so the trajectory is identical and the
    # reported total_wait is the adjusted return.
    adjusted_wait = _greedy_run(_contested_smdp("twin"), targeted_block_policy)["total_wait"]

    # the twin's own greedy no-attack total_wait on the same arrivals is sum_t b(t)
    twin_wait = _greedy_run(_contested_smdp("none"), lambda _s: no_antagonist_policy)["total_wait"]

    assert abs(adjusted_wait - (real_wait - twin_wait)) < 1e-6, (
        adjusted_wait, real_wait, twin_wait)
    # the attack pushes the real episode above the twin, so the adjusted return is a small
    # positive excess rather than the raw wait
    assert twin_wait > 0.0 and adjusted_wait < real_wait


def test_default_path_unchanged():
    # with the field left at its "none" default and no provider attached, the wrapper must
    # reproduce a plain contested run
    a = _greedy_run(_contested_smdp("none"), lambda _s: no_antagonist_policy)["total_wait"]
    plain = SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=RATE, demand_seed=SEED),
        config=contested_config())
    b = run_episode(plain, greedy_insertion_policy(plain), no_antagonist_policy)["total_wait"]
    assert a == b


def test_baseline_series_is_deterministic_and_action_independent():
    # the provider depends only on the arrivals, so repeated calls give an identical series; it
    # runs its own greedy twin and never reads the live agents
    cfg = dataclasses.replace(contested_config(), reward_baseline="twin")
    provider = make_greedy_twin_baseline_provider(cfg, arrival_rate=RATE)
    env1 = make_contested_env(arrival_rate=RATE, demand_seed=SEED)
    env1.reset(demand_seed=SEED)
    env2 = make_contested_env(arrival_rate=RATE, demand_seed=SEED)
    env2.reset(demand_seed=SEED)
    s1, last1 = provider(env1)
    s2, last2 = provider(env2)
    assert s1 == s2 and last1 == last2
    assert len(s1) > 0 and all(v >= 0.0 for v in s1.values())
