#!/usr/bin/env python3
"""Multi-seed, fixed-adversary evaluation for the dynamic assignment rung.

Averages over N fixed Poisson demand instances rather than resting on a single episode, and pits
the learned policy and greedy against the same fixed antagonist on the same instance, so the
reported gap measures robustness rather than arms-race timing. Best-checkpoint selection over the
per-phase snapshots is available too, the final checkpoint being arbitrary under coevolution.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import statistics

import torch

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_dynamic_assign_env
from src.agents.sac import AntagonistSAC, ProtagonistSAC, infer_edge_in_dim, infer_node_in_dim
from src.baselines.greedy_dispatch import greedy_insertion_policy, no_antagonist_policy, run_episode
from scripts.evaluate_assignment import sac_assignment_policy, sac_antagonist_policy


def dynassign_config() -> SMDPConfig:
    """Must match the dynassign branch in scripts/train_sacred.py."""
    return SMDPConfig(
        max_ticks=800, antagonist_interval=25, congestion_duration=120,
        congestion_budget=4000.0, congestion_cooldown=0, congestion_cost=0.1,
        reward_mode="latency", routing_mode="destination", congestion_levels=(1.0,),
        max_antag_actions_per_event=1,
    )


def make_env_for_seed_fn(arrival_rate: float):
    """seed -> a zero-arg factory that builds a dynamic env bound to that fixed demand instance."""
    return lambda seed: (lambda: make_dynamic_assign_env(arrival_rate=arrival_rate, demand_seed=seed))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def eval_dynamic_cells(protag, antag, make_env_for_seed, cfg: SMDPConfig, seeds=(0, 1, 2)) -> dict:
    """Run the four cells of {greedy, learned} x {no attack, attack} over fixed demand seeds.

    Learned and greedy face the same fixed antagonist on each instance. Returns the mean and
    standard deviation per cell plus the gaps; ``gap_atk_mean`` below zero means learned beats
    greedy under the fixed adversary.
    """
    cells: dict[str, list[float]] = {
        "greedy_noatk": [], "greedy_atk": [], "learned_noatk": [], "learned_atk": [],
        "gap_noatk": [], "gap_atk": [],
    }
    for seed in seeds:
        mk = make_env_for_seed(seed)

        def fresh() -> SMDPDecisionWrapper:
            return SMDPDecisionWrapper(env_factory=mk, config=cfg)

        s = fresh(); g_no = run_episode(s, greedy_insertion_policy(s), no_antagonist_policy)["total_wait"]
        s = fresh(); g_at = run_episode(s, greedy_insertion_policy(s), sac_antagonist_policy(s, antag))["total_wait"]
        s = fresh(); l_no = run_episode(s, sac_assignment_policy(protag), no_antagonist_policy)["total_wait"]
        s = fresh(); l_at = run_episode(s, sac_assignment_policy(protag), sac_antagonist_policy(s, antag))["total_wait"]
        cells["greedy_noatk"].append(g_no); cells["greedy_atk"].append(g_at)
        cells["learned_noatk"].append(l_no); cells["learned_atk"].append(l_at)
        cells["gap_noatk"].append(l_no - g_no); cells["gap_atk"].append(l_at - g_at)

    out: dict[str, float] = {}
    for key, vals in cells.items():
        mean, std = _mean_std(vals)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = std
    return out


def _new_protag(cfg: SMDPConfig, node_in_dim: int = 13, edge_in_dim: int = 4) -> ProtagonistSAC:
    return ProtagonistSAC(node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, num_layers=2, heads=4, device="cpu")


def _new_antag(cfg: SMDPConfig, node_in_dim: int = 13, edge_in_dim: int = 4) -> AntagonistSAC:
    return AntagonistSAC(
        node_in_dim=node_in_dim, edge_in_dim=edge_in_dim, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(cfg.congestion_levels),
        level_costs=[lvl * cfg.congestion_duration for lvl in cfg.congestion_levels],
        congestion_levels=cfg.congestion_levels, device="cpu",
    )


def _load_protag(cfg: SMDPConfig, path: str) -> ProtagonistSAC:
    """Build a protagonist sized to the checkpoint's trained feature width, and load it.

    Narrower older checkpoints keep working, because the agent slices current features down to
    its own width.
    """
    sd = torch.load(path, map_location="cpu")
    agent = _new_protag(cfg, node_in_dim=infer_node_in_dim(sd), edge_in_dim=infer_edge_in_dim(sd))
    agent.actor.load_state_dict(sd)
    return agent


def _load_antag(cfg: SMDPConfig, path: str) -> AntagonistSAC:
    sd = torch.load(path, map_location="cpu")
    agent = _new_antag(cfg, node_in_dim=infer_node_in_dim(sd), edge_in_dim=infer_edge_in_dim(sd))
    agent.actor.load_state_dict(sd)
    return agent


def select_best_checkpoint(run_dir, make_env_for_seed, cfg, seeds, antag_path=None) -> list[dict]:
    """Evaluate every protagonist snapshot against a fixed antagonist over the seeds.

    The antagonist defaults to the run's final one, and results are sorted by ``gap_atk_mean``
    ascending. The final checkpoint is arbitrary under coevolution, so the best protagonist under
    a fixed adversary is reported instead, which is also what deployment would ship.
    """
    if antag_path is None:
        antag_path = os.path.join(run_dir, "antagonist", "actor.pt")
    antag = _load_antag(cfg, antag_path)

    snaps = sorted(
        glob.glob(os.path.join(run_dir, "snapshots", "protagonist_ep*.pt")),
        key=lambda p: int(re.search(r"ep(\d+)", p).group(1)),
    )
    if not snaps:  # fall back to the final actor if no per-phase snapshots were saved
        snaps = [os.path.join(run_dir, "protagonist", "actor.pt")]

    results = []
    for path in snaps:
        m = re.search(r"ep(\d+)", path)
        ep = int(m.group(1)) if m else -1
        protag = _load_protag(cfg, path)
        r = eval_dynamic_cells(protag, antag, make_env_for_seed, cfg, seeds)
        results.append({"ep": ep, "path": path, **r})
    results.sort(key=lambda d: d["gap_atk_mean"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-seed, fixed-adversary eval for dynamic assignment.")
    parser.add_argument("--run", required=True, help="models/runs/<run_name>")
    parser.add_argument("--arrival-rate", type=float, default=0.05, help="must match the trained run")
    parser.add_argument("--seeds", type=int, default=5, help="number of fixed Poisson demand instances")
    parser.add_argument("--select-best", action="store_true", help="scan per-phase snapshots, report the best")
    args = parser.parse_args()

    cfg = dynassign_config()
    seeds = tuple(range(args.seeds))
    make_env_for_seed = make_env_for_seed_fn(args.arrival_rate)

    if args.select_best:
        results = select_best_checkpoint(args.run, make_env_for_seed, cfg, seeds)
        print(f"\nBest-checkpoint selection over {len(results)} snapshots "
              f"({len(seeds)} fixed seeds, final antagonist as fixed adversary):")
        for r in results[:12]:
            print(f"  ep{r['ep']:>4}  gap_atk={r['gap_atk_mean']:+7.1f} +/-{r['gap_atk_std']:4.0f}  "
                  f"gap_noatk={r['gap_noatk_mean']:+7.1f}  learned_atk={r['learned_atk_mean']:7.0f}")
        best = results[0]
        verdict = "BEATS" if best["gap_atk_mean"] < 0 else "loses to"
        print(f"\nBEST: ep{best['ep']}  gap_atk={best['gap_atk_mean']:+.1f} +/- {best['gap_atk_std']:.0f}  "
              f"({verdict} greedy under the fixed adversary)")
    else:
        protag = _load_protag(cfg, os.path.join(args.run, "protagonist", "actor.pt"))
        antag = _load_antag(cfg, os.path.join(args.run, "antagonist", "actor.pt"))
        r = eval_dynamic_cells(protag, antag, make_env_for_seed, cfg, seeds)
        print(f"\nDynamic assignment eval - run {args.run} ({len(seeds)} fixed seeds)")
        for cell in ["greedy_noatk", "greedy_atk", "learned_noatk", "learned_atk"]:
            print(f"  {cell:>14}: {r[cell + '_mean']:8.1f} +/- {r[cell + '_std']:5.1f}")
        print(f"  gap_atk   = {r['gap_atk_mean']:+.1f} +/- {r['gap_atk_std']:.0f}  (neg = learned beats greedy under attack)")
        print(f"  gap_noatk = {r['gap_noatk_mean']:+.1f} +/- {r['gap_noatk_std']:.0f}")


if __name__ == "__main__":
    main()
