"""Contested-resupply arena: the shared configuration for its training and evaluation.

Dynamic-assignment physics (Poisson resupply demand, two depots, latency reward,
destination-mode assignment) with the antagonist's reach set to ``route``, so both scripted and
learned attacks aim along each vehicle's committed delivery path. Trainer and evaluation harness
both import ``contested_config()`` from here, so the configuration cannot drift between them.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Callable

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_dynamic_assign_env

# Carried over from the dynamic-assignment configuration.
DEFAULT_CONTESTED_BUDGET = 4000.0


def contested_config(congestion_budget: float = DEFAULT_CONTESTED_BUDGET) -> SMDPConfig:
    """Return the contested-resupply SMDPConfig.

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
        routing_mode="destination",       # assignment only: the env auto-routes
        congestion_levels=(1.0,),         # full blockage only
        max_antag_actions_per_event=1,    # one strategic roadblock per event
        antag_reach="route",              # aim along each vehicle's committed delivery path
    )


# Same geometry and demand as the dynamic-assignment env, re-exported under the arena's name.
make_contested_env = make_dynamic_assign_env


def make_greedy_twin_baseline_provider(
    cfg: SMDPConfig, arrival_rate: float = 0.06,
) -> Callable[[GraphEnv], tuple[dict[int, float], float]]:
    """Build the action-independent reward baseline provider for the contested arena.

    ``provider(real_env) -> (series, last)``, where ``series[t]`` is the outstanding-demand count
    at tick ``t`` of a deterministic greedy, no-attack rollout replaying ``real_env``'s exact
    arrivals, and ``last`` pads ticks the real episode reaches but the twin did not. The baseline
    depends only on the demand realisation and never on the live agents' actions, so subtracting
    it preserves the zero-sum game up to a per-episode constant while removing the arrival trend
    and the latency unavoidable even under a competent clean policy. Costs one greedy rollout
    per episode.
    """
    # Force reward_baseline off so the twin can never recurse into a provider; every other
    # physics knob stays identical to the live arena.
    twin_cfg = replace(cfg, reward_baseline="none")

    def provider(real_env: GraphEnv) -> tuple[dict[int, float], float]:
        # Lazy import avoids an import cycle, since baselines imports env.
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

