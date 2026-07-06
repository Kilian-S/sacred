"""Tests for B1 counterfactual twin reward baseline (Option B), contested arena.

The load-bearing property: subtracting an action-independent per-tick baseline b(t) shifts the
episode return by exactly a per-episode CONSTANT (sum_t b(t) = the twin greedy rollout's
total_wait), preserving the zero-sum game and its equilibrium. Verified numerically end-to-end,
plus the default (reward_baseline="none") path is asserted byte-identical to before.
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
    # attacker_of(smdp) binds the attacker to the SAME wrapper the episode drives (the attacker
    # reads smdp.env each event -- a separate wrapper instance would desync it).
    return run_episode(smdp, greedy_insertion_policy(smdp), attacker_of(smdp))


def test_twin_baseline_shifts_return_by_the_twin_constant():
    # Real episode (greedy vs targeted, no baseline): total_wait_real.
    real_wait = _greedy_run(_contested_smdp("none"), targeted_block_policy)["total_wait"]

    # Same episode WITH the twin baseline: greedy is deterministic and the baseline changes only
    # the reward (not dynamics/actions), so the trajectory is identical -> the reported total_wait
    # is the adjusted return.
    adjusted_wait = _greedy_run(_contested_smdp("twin"), targeted_block_policy)["total_wait"]

    # The twin's own greedy no-attack total_wait on the SAME arrivals = sum_t b(t).
    twin_wait = _greedy_run(_contested_smdp("none"), lambda _s: no_antagonist_policy)["total_wait"]

    # adjusted == real - twin_constant (to floating tolerance).
    assert abs(adjusted_wait - (real_wait - twin_wait)) < 1e-6, (
        adjusted_wait, real_wait, twin_wait)
    # And the subtraction is doing real work (attack made real > twin, so adjusted is a small
    # positive excess, well below the raw real_wait).
    assert twin_wait > 0.0 and adjusted_wait < real_wait


def test_default_path_unchanged():
    # reward_baseline="none" must reproduce a plain contested run (regression guard for historical
    # behaviour: the config field defaults to "none" and no provider is attached).
    a = _greedy_run(_contested_smdp("none"), lambda _s: no_antagonist_policy)["total_wait"]
    plain = SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=RATE, demand_seed=SEED),
        config=contested_config())
    b = run_episode(plain, greedy_insertion_policy(plain), no_antagonist_policy)["total_wait"]
    assert a == b


def test_baseline_series_is_deterministic_and_action_independent():
    # The provider depends only on the arrivals -> identical series on repeated calls, and it does
    # not read the live agents (it runs its own greedy twin).
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
