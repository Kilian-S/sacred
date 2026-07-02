#!/usr/bin/env python3
"""Evaluate a trained assignment-probe protagonist vs greedy-insertion (3b).

Headline check: under the trained antagonist, does the learned multi-truck assignment policy
achieve lower total delivery latency than reactive greedy-insertion? The static cell is
reported too (expected near-0 gap — assignment headroom here is adversarial, see the gate).

Usage:
    PYTHONPATH=. python scripts/evaluate_assignment.py --run models/runs/<run_name>
"""

from __future__ import annotations

import argparse
import os

import torch

from src.env.smdp_wrapper import DecisionEvent, SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_assignment_env
from src.agents.sac import AntagonistSAC, ProtagonistSAC
from src.baselines.greedy_dispatch import greedy_insertion_policy, no_antagonist_policy, run_episode


def assignment_config() -> SMDPConfig:
    """Must match the assign branch in scripts/train_sacred.py."""
    return SMDPConfig(
        max_ticks=800, antagonist_interval=20, congestion_duration=30,
        congestion_budget=400.0, congestion_cooldown=0, congestion_cost=0.1,
        reward_mode="latency", routing_mode="destination", congestion_levels=(0.25, 0.5, 0.75, 1.0),
    )


def sac_assignment_policy(agent: ProtagonistSAC, env_ref=None):
    """Learned multi-truck assignment with STATE PROJECTION + SEQUENTIAL CLAIMING (mirrors the
    trainer): each waiting truck decides in turn; its chosen request is removed from later trucks'
    masks so two trucks can't be assigned the same demand node (depots are never claimed)."""
    def policy(event: DecisionEvent):
        mask = event.protagonist_action_mask
        nodes = event.observation["nodes"]
        is_demand = lambda n: nodes[n].get("demand", 0.0) > 0.0
        actions: dict = {}
        claimed: set = set()
        projected = dict(event.observation)
        projected["trucks"] = {tid: dict(t) for tid, t in event.observation["trucks"].items()}
        for truck_id in event.waiting_trucks:
            truck_mask = {tid: [n for n in opts if n not in claimed] for tid, opts in mask.items()}
            projected["active_truck"] = truck_id
            projected["allowed_destinations"] = {"protagonist": dict(truck_mask)}
            chosen = agent.select_action(projected, truck_mask, deterministic=True)
            actions.update(chosen)
            node = chosen.get(truck_id)
            if node is not None:
                projected["trucks"][truck_id]["destination"] = node
                projected["trucks"][truck_id]["current_node"] = None
                if is_demand(node):
                    claimed.add(node)
        return actions

    return policy


def sac_antagonist_policy(smdp: SMDPDecisionWrapper, agent: AntagonistSAC):
    def policy(event: DecisionEvent):
        return agent.select_action(
            event.observation, event.antagonist_action_mask, smdp.budget.remaining, deterministic=True
        )

    return policy


def eval_cells_assignment(protag, antag, make_env, cfg: SMDPConfig) -> dict:
    """4-cell {greedy-insertion, learned} x {no-attack, attack} eval -> total_wait per cell +
    gaps. Reusable for the CLI and periodic in-training eval. ``gap_atk`` < 0 means the learned
    policy beats greedy-insertion under the trained antagonist (the headline)."""
    def fresh() -> SMDPDecisionWrapper:
        return SMDPDecisionWrapper(env_factory=make_env, config=cfg)

    out: dict = {}
    s = fresh(); out["greedy_noatk"] = run_episode(s, greedy_insertion_policy(s), no_antagonist_policy)["total_wait"]
    s = fresh(); out["greedy_atk"] = run_episode(s, greedy_insertion_policy(s), sac_antagonist_policy(s, antag))["total_wait"]
    s = fresh(); out["learned_noatk"] = run_episode(s, sac_assignment_policy(protag), no_antagonist_policy)["total_wait"]
    s = fresh(); out["learned_atk"] = run_episode(s, sac_assignment_policy(protag), sac_antagonist_policy(s, antag))["total_wait"]
    out["gap_atk"] = out["learned_atk"] - out["greedy_atk"]
    out["gap_noatk"] = out["learned_noatk"] - out["greedy_noatk"]
    return out


def _build_agents(run_dir: str, cfg: SMDPConfig):
    protag = ProtagonistSAC(node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4, device="cpu")
    antag = AntagonistSAC(
        node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(cfg.congestion_levels),
        level_costs=[lvl * cfg.congestion_duration for lvl in cfg.congestion_levels], device="cpu",
    )
    protag.actor.load_state_dict(torch.load(os.path.join(run_dir, "protagonist", "actor.pt"), map_location="cpu"))
    antag.actor.load_state_dict(torch.load(os.path.join(run_dir, "antagonist", "actor.pt"), map_location="cpu"))
    return protag, antag


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained assignment protagonist vs greedy-insertion.")
    parser.add_argument("--run", required=True, help="models/runs/<run_name> with actor.pt files")
    args = parser.parse_args()

    cfg = assignment_config()
    protag, antag = _build_agents(args.run, cfg)
    r = eval_cells_assignment(protag, antag, lambda: make_assignment_env(), cfg)

    print(f"\nAssignment evaluation — run: {args.run}")
    print(f"{'policy':>8} | {'antagonist':>10} | {'total_wait':>10}")
    print("-" * 36)
    print(f"{'greedy':>8} | {'no-attack':>10} | {r['greedy_noatk']:>10.1f}")
    print(f"{'greedy':>8} | {'attack':>10} | {r['greedy_atk']:>10.1f}")
    print(f"{'learned':>8} | {'no-attack':>10} | {r['learned_noatk']:>10.1f}")
    print(f"{'learned':>8} | {'attack':>10} | {r['learned_atk']:>10.1f}")
    print("\n--- headline (under trained antagonist) ---")
    print(f"greedy={r['greedy_atk']:.0f} learned={r['learned_atk']:.0f} gap={r['gap_atk']:+.0f}  "
          f"(neg = learned beats greedy-insertion under attack)")
    if r["gap_atk"] < 0:
        print(f"PASS: learned beats greedy-insertion by {-r['gap_atk']:.0f} "
              f"({-100*r['gap_atk']/r['greedy_atk']:.1f}%) under attack.")
    else:
        print("NOT BEATEN under attack — inspect Q_Spread / entropy / antagonist co-evolution.")


if __name__ == "__main__":
    main()
