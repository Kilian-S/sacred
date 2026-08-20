"""Contested-resupply arena: the dynamic-assignment dynamics (Poisson resupply demand, two
depots, latency reward) with the antagonist's reach set to ``route``, so attacks aim along each
vehicle's committed delivery path. Both the trainer and the evaluation harness import
``contested_config()`` from here so the train and eval configs cannot drift apart.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_dynamic_assign_env

DEFAULT_CONTESTED_BUDGET = 4000.0


def contested_config(congestion_budget: float = DEFAULT_CONTESTED_BUDGET) -> SMDPConfig:
    """The contested-resupply SMDPConfig.

    Kept in lockstep with ``scripts/evaluate_dynamic_assign.dynassign_config`` except for
    ``antag_reach="route"``.
    """
    return SMDPConfig(
        max_ticks=800,
        antagonist_interval=25,           # ~32 antagonist decision events / episode
        congestion_duration=120,          # each full roadblock persists ~5 events (sustained)
        congestion_budget=congestion_budget,
        congestion_cooldown=0,
        congestion_cost=0.1,
        reward_mode="latency",
        routing_mode="destination",       # assignment only: env auto-routes (routing deferred)
        congestion_levels=(1.0,),         # full blockage only
        max_antag_actions_per_event=1,    # one strategic roadblock per event
        antag_reach="route",              # aim along each vehicle's committed delivery path
    )


# Same env as the dynamic-assignment arena, re-exported under the contested-framing name.
make_contested_env = make_dynamic_assign_env


def make_greedy_twin_baseline_provider(
    cfg: SMDPConfig, arrival_rate: float = 0.06,
) -> Callable[[GraphEnv], tuple[dict[int, float], float]]:
    """Build a provider of the action-independent latency baseline b(t).

    The baseline is the outstanding-demand curve of a deterministic greedy, no-attack rollout
    replaying the live episode's exact arrivals. It depends only on the demand realisation, never
    on the agents' actions, so subtracting it preserves the zero-sum game up to a per-episode
    constant. Costs one greedy rollout per episode.

    Returns:
        ``provider(real_env) -> (series, last)``, where ``series`` maps tick to outstanding
        demand and ``last`` pads ticks the real episode reaches but the clean twin did not.
    """
    # reward_baseline off so the twin can never recurse into a provider
    twin_cfg = replace(cfg, reward_baseline="none")

    def provider(real_env: GraphEnv) -> tuple[dict[int, float], float]:
        # lazy import: baselines imports env, so a module-level import would cycle
        from src.baselines.greedy_dispatch import (
            greedy_insertion_policy, no_antagonist_policy, run_episode)

        schedule = list(getattr(real_env, "_arrival_schedule", []))
        twin = SMDPDecisionWrapper(
            env_factory=lambda: make_contested_env(
                arrival_rate=arrival_rate, arrival_schedule=schedule),
            config=twin_cfg,
        )
        twin._baseline_record = []  # capture per-tick remaining_demand during the rollout
        run_episode(twin, greedy_insertion_policy(twin), no_antagonist_policy)
        series = dict(twin._baseline_record)  # tick -> outstanding demand (last write per tick wins)
        last = series[max(series)] if series else 0.0
        return series, last

    return provider

