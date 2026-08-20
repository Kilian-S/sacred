#!/usr/bin/env python3
"""Evaluation for the hybrid rung (assignment plus next-hop routing): the learned policy against
the hybrid greedy baseline, with and without a fixed antagonist. Demand is static, so each cell is
a single deterministic episode and best-checkpoint selection carries no max-over-noise bias.

  PYTHONPATH=. python scripts/evaluate_hybrid.py --run models/runs/<run>
  PYTHONPATH=. python scripts/evaluate_hybrid.py --run models/runs/<run> --select-best
"""

from __future__ import annotations

import argparse
import glob
import os
import re

import torch

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_hybrid_assign_env
from src.agents.sac import AntagonistSAC, ProtagonistSAC, infer_edge_in_dim, infer_node_in_dim
from src.baselines.greedy_dispatch import hybrid_greedy_policy, no_antagonist_policy, run_episode
from scripts.evaluate_assignment import sac_antagonist_policy


def hybrid_config() -> SMDPConfig:
    """SMDP configuration for the hybrid rung."""
    return SMDPConfig(
        max_ticks=800, reward_mode="latency", routing_mode="hybrid", routing_corridor_slack=2.0,
        antagonist_interval=25, congestion_duration=125, congestion_budget=1500.0, congestion_cooldown=0,
        congestion_cost=0.1, congestion_levels=(1.0,), max_antag_actions_per_event=1, antag_reach="route")


def sac_hybrid_policy(smdp: SMDPDecisionWrapper, agent: ProtagonistSAC):
    """Learned hybrid policy: one action per waiting truck, over assignment candidates or routing
    next-hops depending on ``assigned_target``. Only assignment trucks claim a demand node."""
    def policy(event):
        env = smdp.env
        mask = event.protagonist_action_mask
        demand_nodes = set(getattr(env, "assignment_demand", ()))
        actions, claimed = {}, set()
        projected = dict(event.observation)
        projected["trucks"] = {tid: dict(t) for tid, t in event.observation["trucks"].items()}
        for tid in sorted(mask):
            truck = env.trucks[tid]
            is_assign = truck.assigned_target is None
            opts = [n for n in mask[tid] if (not is_assign or n not in claimed)]
            if not opts:
                continue
            truck_mask = {tid: opts}
            projected["active_truck"] = tid
            projected["allowed_destinations"] = {"protagonist": dict(truck_mask)}
            chosen = agent.select_action(projected, truck_mask, deterministic=True)
            actions.update(chosen)
            node = chosen.get(tid)
            if is_assign and node in demand_nodes:
                claimed.add(node)
        return actions
    return policy


def eval_hybrid_cells(protag, antag, make_env, cfg: SMDPConfig) -> dict:
    """Run the four-cell {greedy, learned} x {no-attack, fixed antagonist} evaluation.

    Returns:
        Total wait per cell plus the two gaps. ``gap_atk`` < 0 means the learned policy beats the
        hybrid greedy baseline under the fixed adversary.
    """
    def fresh() -> SMDPDecisionWrapper:
        return SMDPDecisionWrapper(env_factory=make_env, config=cfg)

    out: dict = {}
    s = fresh(); out["greedy_noatk"] = run_episode(s, hybrid_greedy_policy(s), no_antagonist_policy)["total_wait"]
    s = fresh(); out["greedy_atk"] = run_episode(s, hybrid_greedy_policy(s), sac_antagonist_policy(s, antag))["total_wait"]
    s = fresh(); out["learned_noatk"] = run_episode(s, sac_hybrid_policy(s, protag), no_antagonist_policy)["total_wait"]
    s = fresh(); out["learned_atk"] = run_episode(s, sac_hybrid_policy(s, protag), sac_antagonist_policy(s, antag))["total_wait"]
    out["gap_atk"] = out["learned_atk"] - out["greedy_atk"]
    out["gap_noatk"] = out["learned_noatk"] - out["greedy_noatk"]
    return out


def _new_protag(node_in_dim: int = 13, edge_in_dim: int = 4) -> ProtagonistSAC:
    return ProtagonistSAC(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, num_layers=2, heads=4, device="cpu")


def _new_antag(cfg: SMDPConfig, node_in_dim: int = 13, edge_in_dim: int = 4) -> AntagonistSAC:
    return AntagonistSAC(
        node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(cfg.congestion_levels),
        level_costs=[lvl * cfg.congestion_duration for lvl in cfg.congestion_levels],
        congestion_levels=cfg.congestion_levels, device="cpu")


def _load_protag(path) -> ProtagonistSAC:
    """Load a protagonist, sizing the nets to the checkpoint's trained feature width."""
    sd = torch.load(path, map_location="cpu")
    agent = _new_protag(node_in_dim=infer_node_in_dim(sd), edge_in_dim=infer_edge_in_dim(sd))
    agent.actor.load_state_dict(sd)
    return agent


def _load_antag(cfg: SMDPConfig, path) -> AntagonistSAC:
    sd = torch.load(path, map_location="cpu")
    agent = _new_antag(cfg, node_in_dim=infer_node_in_dim(sd), edge_in_dim=infer_edge_in_dim(sd))
    agent.actor.load_state_dict(sd)
    return agent


def select_best_checkpoint(run_dir, make_env, cfg, antag_path=None) -> list[dict]:
    """Evaluate every protagonist snapshot against a fixed antagonist.

    Returns:
        Per-snapshot results sorted best-first, that is by most negative ``gap_atk``.
    """
    if antag_path is None:
        antag_path = os.path.join(run_dir, "antagonist", "actor.pt")
    antag = _load_antag(cfg, antag_path)
    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshots", "protagonist_ep*.pt")),
                   key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    if not snaps:
        snaps = [os.path.join(run_dir, "protagonist", "actor.pt")]
    results = []
    for path in snaps:
        m = re.search(r"ep(\d+)", path)
        ep = int(m.group(1)) if m else -1
        protag = _load_protag(path)
        r = eval_hybrid_cells(protag, antag, make_env, cfg)
        results.append({"ep": ep, **r})
    results.sort(key=lambda d: d["gap_atk"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained hybrid protagonist vs the hybrid greedy baseline.")
    parser.add_argument("--run", required=True, help="models/runs/<run_name>")
    parser.add_argument("--select-best", action="store_true", help="scan per-phase snapshots, report the best")
    args = parser.parse_args()

    cfg = hybrid_config()
    make_env = make_hybrid_assign_env

    if args.select_best:
        results = select_best_checkpoint(args.run, make_env, cfg)
        print(f"\nBest-checkpoint selection over {len(results)} snapshots (fixed final antagonist):")
        for r in results[:12]:
            print(f"  ep{r['ep']:>4}  gap_atk={r['gap_atk']:+8.0f}  gap_noatk={r['gap_noatk']:+8.0f}  "
                  f"learned_atk={r['learned_atk']:8.0f}  greedy_atk={r['greedy_atk']:8.0f}")
        best = results[0]
        verdict = "BEATS" if best["gap_atk"] < 0 else "loses to"
        print(f"\nBEST: ep{best['ep']}  gap_atk={best['gap_atk']:+.0f}  ({verdict} greedy under the fixed adversary)")
    else:
        protag = _load_protag(os.path.join(args.run, "protagonist", "actor.pt"))
        antag = _load_antag(cfg, os.path.join(args.run, "antagonist", "actor.pt"))
        r = eval_hybrid_cells(protag, antag, make_env, cfg)
        print(f"\nHybrid eval — run {args.run}")
        for cell in ["greedy_noatk", "greedy_atk", "learned_noatk", "learned_atk"]:
            print(f"  {cell:>14}: {r[cell]:8.0f}")
        print(f"  gap_atk   = {r['gap_atk']:+.0f}  (neg = learned beats greedy under attack)")
        print(f"  gap_noatk = {r['gap_noatk']:+.0f}")


if __name__ == "__main__":
    main()
