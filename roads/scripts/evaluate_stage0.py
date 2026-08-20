#!/usr/bin/env python3
"""Evaluate a trained Stage-0 protagonist against the greedy baseline.

Checks whether the learned protagonist achieves lower total delivery latency (``total_wait``)
than the greedy nearest-request dispatcher under the trained antagonist, with the no-attack
column reported alongside to show the antagonist's bite.
"""

from __future__ import annotations

import argparse
import os

import torch

from src.env.smdp_wrapper import DecisionEvent, SMDPConfig, SMDPDecisionWrapper
from src.envs.stage0_factory import make_stage0_nexthop_env
from src.agents.sac import AntagonistSAC, ProtagonistSAC
from src.baselines.greedy_dispatch import (
    greedy_next_hop_policy,
    no_antagonist_policy,
    run_episode,
)


def stage0_config() -> SMDPConfig:
    """Must match the Stage-0 config in scripts/train_sacred.py."""
    return SMDPConfig(
        max_ticks=400,
        antagonist_interval=20,
        congestion_duration=30,
        congestion_budget=300.0,
        congestion_cooldown=0,
        congestion_cost=0.1,
        reward_mode="latency",
        routing_mode="next_hop",
        routing_corridor_slack=1.2,  # must match train_sacred.py; the 1.5 default was not trained
        congestion_levels=(0.25, 0.5, 0.75, 1.0),
    )


def sac_protagonist_policy(agent: ProtagonistSAC):
    def policy(event: DecisionEvent):
        mask = event.protagonist_action_mask
        actions: dict = {}
        for truck_id in event.waiting_trucks:
            obs = dict(event.observation)
            obs["active_truck"] = truck_id
            obs["allowed_destinations"] = {"protagonist": dict(mask)}
            actions.update(agent.select_action(obs, mask, deterministic=True))
        return actions

    return policy


def sac_antagonist_policy(smdp: SMDPDecisionWrapper, agent: AntagonistSAC):
    def policy(event: DecisionEvent):
        return agent.select_action(
            event.observation, event.antagonist_action_mask, smdp.budget.remaining, deterministic=True
        )

    return policy


def eval_cells(protag, antag, make_env, cfg: SMDPConfig) -> dict:
    """Run the four cells of {greedy, learned} x {no attack, attack}.

    Returns total_wait per cell plus the headline gap, and serves both the command line and
    periodic in-training evaluation. Deterministic, since neither env nor policies introduce
    randomness. ``gap_atk`` below zero means the learned policy beats greedy under the antagonist.
    """
    def fresh() -> SMDPDecisionWrapper:
        return SMDPDecisionWrapper(env_factory=make_env, config=cfg)

    out: dict = {}
    s = fresh(); out["greedy_noatk"] = run_episode(s, greedy_next_hop_policy(s), no_antagonist_policy)["total_wait"]
    s = fresh(); out["greedy_atk"] = run_episode(s, greedy_next_hop_policy(s), sac_antagonist_policy(s, antag))["total_wait"]
    s = fresh(); out["learned_noatk"] = run_episode(s, sac_protagonist_policy(protag), no_antagonist_policy)["total_wait"]
    s = fresh(); out["learned_atk"] = run_episode(s, sac_protagonist_policy(protag), sac_antagonist_policy(s, antag))["total_wait"]
    out["gap_atk"] = out["learned_atk"] - out["greedy_atk"]
    return out


def _build_agents(run_dir: str, cfg: SMDPConfig):
    protag = ProtagonistSAC(
        node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4, device="cpu"
    )
    antag = AntagonistSAC(
        node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(cfg.congestion_levels),
        level_costs=[lvl * cfg.congestion_duration for lvl in cfg.congestion_levels],
        device="cpu",
    )
    protag.actor.load_state_dict(torch.load(os.path.join(run_dir, "protagonist", "actor.pt"), map_location="cpu"))
    antag.actor.load_state_dict(torch.load(os.path.join(run_dir, "antagonist", "actor.pt"), map_location="cpu"))
    return protag, antag


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Stage-0 protagonist vs greedy.")
    parser.add_argument("--run", required=True, help="models/runs/<run_name> directory with actor.pt files")
    args = parser.parse_args()

    cfg = stage0_config()
    protag, antag = _build_agents(args.run, cfg)

    def fresh_smdp() -> SMDPDecisionWrapper:
        return SMDPDecisionWrapper(env_factory=lambda: make_stage0_nexthop_env(), config=cfg)

    # Four cells: {greedy, learned} x {no-attack, trained antagonist}.
    smdp = fresh_smdp()
    rows = {}
    rows[("greedy", "no-attack")] = run_episode(smdp := fresh_smdp(), greedy_next_hop_policy(smdp), no_antagonist_policy)
    rows[("greedy", "attack")] = run_episode(smdp := fresh_smdp(), greedy_next_hop_policy(smdp), sac_antagonist_policy(smdp, antag))
    rows[("learned", "no-attack")] = run_episode(smdp := fresh_smdp(), sac_protagonist_policy(protag), no_antagonist_policy)
    rows[("learned", "attack")] = run_episode(smdp := fresh_smdp(), sac_protagonist_policy(protag), sac_antagonist_policy(smdp, antag))

    print(f"\nStage-0 evaluation — run: {args.run}")
    print(f"{'policy':>8} | {'antagonist':>10} | {'total_wait':>10} | {'delivered':>9} | {'ticks':>5} | {'budget':>6}")
    print("-" * 64)
    for (pol, atk), r in rows.items():
        print(f"{pol:>8} | {atk:>10} | {r['total_wait']:>10.1f} | {r['delivered']:>3d}/{r['num_requests']:<5d} | {r['ticks']:>5d} | {r['budget_used']:>6.0f}")

    gw = rows[("greedy", "attack")]["total_wait"]
    lw = rows[("learned", "attack")]["total_wait"]
    print("\n--- Stage-0 headline (under trained antagonist) ---")
    print(f"greedy total_wait = {gw:.1f}   learned total_wait = {lw:.1f}   delta = {lw - gw:+.1f}")
    if lw < gw:
        print(f"PASS: learned beats greedy by {gw - lw:.1f} ({100*(gw-lw)/gw:.1f}%) lower latency under attack.")
    else:
        print("NOT BEATEN: learned does not beat greedy under attack — inspect Q_Spread/entropy curves.")


if __name__ == "__main__":
    main()
