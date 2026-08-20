#!/usr/bin/env python3
"""Robustness-portfolio evaluation — the headline harness for the reframed thesis claim.

For each protagonist ARM (greedy reference + learned checkpoints) and each ATTACKER in a
portfolio, runs paired episodes and reports:

  W(arm, attack)               mean total_wait (lower = better)
  D(arm, attack)               degradation = W(attack) - W(none), paired per instance
  dD(attack) = D(vanilla) - D(sacred)   the PRIMARY metric: > 0 => adversarial training bought
                                        robustness (computed for every learned-arm pair)

Design (per CRITIQUE.md §2/§6):
  * every arm faces the SAME attacker set on the SAME instances (paired, common random numbers);
  * learned protagonists act STOCHASTICALLY (they are max-entropy policies; argmax-ing them
    destroys the mixed strategies adversarial training may have bought), seeded per episode for
    reproducibility;
  * attackers: none / random (seeded) / targeted (scripted heuristic) / br_<arm> (learned
    best-response nets, trained per-policy via --train-antagonist-only);
  * checkpoint selection (--select-best) happens on the VALIDATION attacker (targeted) over
    VALIDATION instances; the best-response attackers + test instances stay held out;
  * dynassign test instances start at seed 10_000_019 (validation at 20_000_019) — far from any
    training run's demand-seed counter (seed*100003 + episode), so eval streams are held out.

Usage (Phase 1, dynassign):
  PYTHONPATH=. python scripts/evaluate_portfolio.py --problem dynassign \
      --policy sacred=models/runs/gen02_dynassign/dynassign_seed0/snapshots/protagonist_ep550.pt \
      --policy vanilla=models/runs/gen03_vanilla/vanilla_seed0/snapshots/protagonist_ep550.pt \
      --br sacred=models/runs/gen03_br/br_sacred_seed0/antagonist/actor.pt \
      --br vanilla=models/runs/gen03_br/br_vanilla_seed0/antagonist/actor.pt \
      --instances 30 --out experiments/gen03_portfolio_seed0.json

  PYTHONPATH=. python scripts/evaluate_portfolio.py --problem dynassign \
      --select-best models/runs/gen03_vanilla/vanilla_seed0 --instances 8
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from typing import Any, Callable

import torch

from src.env.smdp_wrapper import DecisionEvent, SMDPConfig, SMDPDecisionWrapper
from src.agents.sac import AntagonistSAC, ProtagonistSAC, infer_edge_in_dim, infer_node_in_dim
from src.baselines.attackers import mask_first_block_policy, random_block_policy, random_path_block_policy, targeted_block_policy
from src.baselines.greedy_dispatch import (
    greedy_insertion_policy,
    hybrid_greedy_policy,
    no_antagonist_policy,
    run_episode,
)

# Held-out instance-seed bases (see module docstring).
TEST_SEED_BASE = 10_000_019
VAL_SEED_BASE = 20_000_019


# ---------------------------------------------------------------------------
# Problem plumbing


def _problem_setup(problem: str, arrival_rate: float):
    """Return (cfg, make_env_for_seed, greedy_factory, is_static) for a problem."""
    if problem == "dynassign":
        from scripts.evaluate_dynamic_assign import dynassign_config
        from src.envs.assignment_factory import make_dynamic_assign_env

        cfg = dynassign_config()
        mk = lambda seed: (lambda: make_dynamic_assign_env(arrival_rate=arrival_rate, demand_seed=seed))
        return cfg, mk, greedy_insertion_policy, False
    if problem == "hybrid":
        from scripts.evaluate_hybrid import hybrid_config
        from src.envs.assignment_factory import make_hybrid_assign_env

        cfg = hybrid_config()
        mk = lambda seed: make_hybrid_assign_env  # static demand: the seed only drives rollouts
        return cfg, mk, hybrid_greedy_policy, True
    if problem == "contested":
        # gen07 arena: dynassign dynamics + route reach (single source of truth in contested.py).
        from src.envs.contested import contested_config, make_contested_env

        cfg = contested_config()
        mk = lambda seed: (lambda: make_contested_env(arrival_rate=arrival_rate, demand_seed=seed))
        return cfg, mk, greedy_insertion_policy, False
    raise ValueError(f"unsupported problem {problem!r}")


def _load_protagonist(path: str) -> ProtagonistSAC:
    sd = torch.load(path, map_location="cpu")
    agent = ProtagonistSAC(node_in_dim=infer_node_in_dim(sd), edge_in_dim=infer_edge_in_dim(sd),
                           hidden_dim=64, num_layers=2, heads=4, device="cpu")
    agent.actor.load_state_dict(sd)
    return agent


def _load_antagonist(path: str, cfg: SMDPConfig) -> AntagonistSAC:
    sd = torch.load(path, map_location="cpu")
    agent = AntagonistSAC(
        node_in_dim=infer_node_in_dim(sd), edge_in_dim=infer_edge_in_dim(sd),
        hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(cfg.congestion_levels),
        level_costs=[lvl * cfg.congestion_duration for lvl in cfg.congestion_levels],
        congestion_levels=cfg.congestion_levels, device="cpu")
    agent.actor.load_state_dict(sd)
    return agent


# ---------------------------------------------------------------------------
# Policies


def sac_protagonist_policy(smdp: SMDPDecisionWrapper, agent: ProtagonistSAC, *,
                           deterministic: bool = False):
    """Learned protagonist for EITHER decision model, with state projection + sequential claiming
    (mirrors the trainer/prior eval policies). Handles destination-mode assignment and hybrid
    (assignment vs routing branches by ``assigned_target``); stochastic by default — the eval
    counterpart of the max-entropy policy actually trained."""

    def policy(event: DecisionEvent):
        env = smdp.env
        mask = event.protagonist_action_mask
        nodes = event.observation["nodes"]
        actions: dict = {}
        claimed: set = set()
        projected = dict(event.observation)
        projected["trucks"] = {tid: dict(t) for tid, t in event.observation["trucks"].items()}
        for tid in sorted(mask):
            truck = env.trucks[tid]
            # In hybrid mode a truck with an assigned target is ROUTING (no claiming); everything
            # else is an assignment/destination decision subject to claiming.
            is_routing = smdp.config.routing_mode == "hybrid" and truck.assigned_target is not None
            opts = [n for n in mask[tid] if (is_routing or n not in claimed)]
            if not opts:
                continue
            truck_mask = {tid: opts}
            projected["active_truck"] = tid
            projected["allowed_destinations"] = {"protagonist": dict(truck_mask)}
            chosen = agent.select_action(projected, truck_mask, deterministic=deterministic)
            actions.update(chosen)
            node = chosen.get(tid)
            if node is not None:
                projected["trucks"][tid]["destination"] = node
                projected["trucks"][tid]["current_node"] = None
                if not is_routing and nodes[node].get("demand", 0.0) > 0.0:
                    claimed.add(node)
        return actions

    return policy


def _sac_attacker(smdp: SMDPDecisionWrapper, agent: AntagonistSAC):
    """Learned (best-response) attacker: deterministic — its strongest single response."""
    def policy(event: DecisionEvent):
        return agent.select_action(
            event.observation, event.antagonist_action_mask, smdp.budget.remaining,
            deterministic=True)
    return policy


def _make_attacker(name: str, smdp: SMDPDecisionWrapper, instance_seed: int,
                   br_agents: dict[str, AntagonistSAC]):
    if name == "none":
        return no_antagonist_policy
    if name == "random":
        return random_block_policy(seed=instance_seed)
    if name == "targeted":
        return targeted_block_policy(smdp)
    if name == "pathrand":
        # gen06 training attacker (in-distribution row for the scripted arm; seeded per instance)
        return random_path_block_policy(smdp, seed=instance_seed)
    if name == "gateway":
        # first-maskable-edge attacker — the HELD-OUT strong attack for the hybrid matrix
        # (under route reach the mask does the aiming; +40..184% on greedy in the budget sweep)
        return mask_first_block_policy
    if name.startswith("br_"):
        return _sac_attacker(smdp, br_agents[name[3:]])
    raise ValueError(f"unknown attacker {name!r}")


# ---------------------------------------------------------------------------
# The matrix


def _episode_seed(arm: str, attacker: str, instance_seed: int, rollout: int) -> int:
    # zlib.crc32 is stable across processes (str hash() is salted -> irreproducible).
    import zlib
    return zlib.crc32(f"{arm}|{attacker}|{instance_seed}|{rollout}".encode()) % (2**31 - 1)


def run_matrix(
    policy_factories: dict[str, Callable[[SMDPDecisionWrapper], Any]],
    attacker_names: list[str],
    br_agents: dict[str, AntagonistSAC],
    cfg: SMDPConfig,
    make_env_for_seed,
    instance_seeds: list[int],
    rollouts: int = 1,
    quiet: bool = False,
) -> dict[str, dict[str, list[float]]]:
    """results[arm][attacker] = per-(instance,rollout) total_wait, paired across arms/attackers."""
    results: dict[str, dict[str, list[float]]] = {
        arm: {a: [] for a in attacker_names} for arm in policy_factories
    }
    for inst in instance_seeds:
        mk = make_env_for_seed(inst)
        for arm, factory in policy_factories.items():
            for attacker in attacker_names:
                for r in range(rollouts):
                    smdp = SMDPDecisionWrapper(env_factory=mk, config=cfg)
                    torch.manual_seed(_episode_seed(arm, attacker, inst, r))
                    w = run_episode(smdp, factory(smdp),
                                    _make_attacker(attacker, smdp, inst, br_agents))["total_wait"]
                    results[arm][attacker].append(w)
        if not quiet:
            done = sum(len(v) for a in results.values() for v in a.values())
            print(f"  instance {inst}: done ({done} episodes total)")
    return results


def _mean_sem(xs: list[float]) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0] if xs else float("nan")), 0.0
    return statistics.mean(xs), statistics.stdev(xs) / math.sqrt(len(xs))


def summarize(results: dict[str, dict[str, list[float]]], reference_arm: str = "greedy") -> dict:
    """W / D tables + the paired-primary dD for every learned-arm pair."""
    out: dict[str, Any] = {"W": {}, "D": {}, "dD": {}}
    for arm, per_attack in results.items():
        out["W"][arm] = {a: _mean_sem(v) for a, v in per_attack.items()}
        if "none" in per_attack:
            base = per_attack["none"]
            out["D"][arm] = {
                a: _mean_sem([w - b for w, b in zip(v, base)])
                for a, v in per_attack.items() if a != "none"
            }
    learned = [a for a in results if a != reference_arm]
    for i, arm_a in enumerate(learned):
        for arm_b in learned[i + 1:]:
            # dD > 0 means arm_b degrades MORE than arm_a under this attack (arm_a more robust).
            key = f"{arm_b}-minus-{arm_a}"
            out["dD"][key] = {}
            for attack in results[arm_a]:
                if attack == "none":
                    continue
                da = [w - b for w, b in zip(results[arm_a][attack], results[arm_a]["none"])]
                db = [w - b for w, b in zip(results[arm_b][attack], results[arm_b]["none"])]
                diffs = [y - x for x, y in zip(da, db)]
                out["dD"][key][attack] = _mean_sem(diffs)
    return out


def print_summary(summary: dict) -> None:
    print("\n=== W(arm, attack): mean total_wait ± SEM (lower = better) ===")
    attacks = None
    for arm, per in summary["W"].items():
        if attacks is None:
            attacks = list(per)
            print(f"{'arm':>10} | " + " | ".join(f"{a:>18}" for a in attacks))
        print(f"{arm:>10} | " + " | ".join(f"{per[a][0]:9.0f} ±{per[a][1]:6.0f}" for a in attacks))
    print("\n=== D(arm, attack) = W(attack) − W(none): degradation under attack ===")
    for arm, per in summary["D"].items():
        print(f"{arm:>10} | " + " | ".join(f"{a}: {m:+8.0f} ±{s:5.0f}" for a, (m, s) in per.items()))
    if summary["dD"]:
        print("\n=== PRIMARY: dD ± SEM (positive => the FIRST-named arm degrades more) ===")
        for pair, per in summary["dD"].items():
            for attack, (m, s) in per.items():
                ci = 1.96 * s
                sig = "SIGNIFICANT" if abs(m) > ci and s > 0 else "not significant"
                print(f"  {pair:>28} under {attack:>12}: {m:+8.0f} ± {ci:6.0f} (95% CI) [{sig}]")


# ---------------------------------------------------------------------------
# Checkpoint selection (validation attacker + validation instances only)


def select_best_under_attack(run_dir: str, cfg: SMDPConfig, make_env_for_seed,
                             greedy_factory, instance_seeds: list[int],
                             rollouts: int = 1, attacker: str = "targeted") -> list[dict]:
    """Rank a run's protagonist snapshots by mean total_wait under the VALIDATION attacker
    (configurable — gen06 selects on `pathrand` because `targeted` is its held-out test attack).
    Selection never sees the test attackers or the test instances."""
    import glob
    import os
    import re

    snaps = sorted(glob.glob(os.path.join(run_dir, "snapshots", "protagonist_ep*.pt")),
                   key=lambda p: int(re.search(r"ep(\d+)", p).group(1)))
    if not snaps:
        snaps = [os.path.join(run_dir, "protagonist", "actor.pt")]
    ranked = []
    for path in snaps:
        m = re.search(r"ep(\d+)", path)
        ep = int(m.group(1)) if m else -1
        agent = _load_protagonist(path)
        factory = lambda smdp, _a=agent: sac_protagonist_policy(smdp, _a)
        res = run_matrix({"cand": factory}, [attacker], {}, cfg, make_env_for_seed,
                         instance_seeds, rollouts, quiet=True)
        mean, sem = _mean_sem(res["cand"][attacker])
        ranked.append({"ep": ep, "path": path, "val_attacked_wait": mean, "sem": sem})
    ranked.sort(key=lambda d: d["val_attacked_wait"])
    return ranked


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Robustness-portfolio evaluation.")
    parser.add_argument("--problem", choices=["dynassign", "hybrid", "contested"], default="dynassign")
    parser.add_argument("--arrival-rate", type=float, default=0.06, help="dynassign only")
    parser.add_argument("--policy", action="append", default=[], metavar="NAME=ACTOR.PT",
                        help="learned arm (repeatable); greedy is always included")
    parser.add_argument("--br", action="append", default=[], metavar="NAME=ANTAG_ACTOR.PT",
                        help="best-response antagonist for arm NAME -> attacker 'br_NAME'")
    parser.add_argument("--attackers", type=str, default=None,
                        help="comma list; default: none,random,targeted + all br_<name>")
    parser.add_argument("--instances", type=int, default=30)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--seed-base", type=int, default=TEST_SEED_BASE)
    parser.add_argument("--out", type=str, default=None, help="write raw results + summary JSON")
    parser.add_argument("--select-best", type=str, default=None, metavar="RUN_DIR",
                        help="rank RUN_DIR's snapshots under the validation attacker instead")
    parser.add_argument("--select-attacker", type=str, default="targeted",
                        help="validation attacker for --select-best (gen06: pathrand)")
    args = parser.parse_args()

    cfg, make_env_for_seed, greedy_factory, is_static = _problem_setup(args.problem, args.arrival_rate)

    if args.select_best:
        seeds = [VAL_SEED_BASE + i for i in range(args.instances)]
        ranked = select_best_under_attack(args.select_best, cfg, make_env_for_seed,
                                          greedy_factory, seeds, args.rollouts,
                                          attacker=args.select_attacker)
        print(f"\nSnapshot ranking under the VALIDATION ({args.select_attacker}) attacker "
              f"({args.instances} validation instances):")
        for r in ranked[:15]:
            print(f"  ep{r['ep']:>5}  attacked_wait={r['val_attacked_wait']:8.0f} ±{r['sem']:5.0f}")
        print(f"\nBEST: ep{ranked[0]['ep']}  ({ranked[0]['path']})")
        return

    policy_factories: dict[str, Any] = {"greedy": greedy_factory}
    for spec in args.policy:
        name, path = spec.split("=", 1)
        agent = _load_protagonist(path)
        policy_factories[name] = lambda smdp, _a=agent: sac_protagonist_policy(smdp, _a)

    br_agents: dict[str, AntagonistSAC] = {}
    for spec in args.br:
        name, path = spec.split("=", 1)
        br_agents[name] = _load_antagonist(path, cfg)

    if args.attackers:
        attacker_names = args.attackers.split(",")
    else:
        attacker_names = ["none", "random", "targeted"] + [f"br_{n}" for n in br_agents]

    seeds = [args.seed_base + i for i in range(args.instances)]
    total = len(policy_factories) * len(attacker_names) * len(seeds) * args.rollouts
    print(f"Portfolio: arms={list(policy_factories)} attackers={attacker_names} "
          f"instances={len(seeds)} rollouts={args.rollouts} -> {total} episodes")

    results = run_matrix(policy_factories, attacker_names, br_agents, cfg,
                         make_env_for_seed, seeds, args.rollouts)
    summary = summarize(results)
    print_summary(summary)

    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"results": results, "summary": summary,
                       "config": {"problem": args.problem, "instances": args.instances,
                                  "rollouts": args.rollouts, "seed_base": args.seed_base,
                                  "attackers": attacker_names,
                                  "policies": args.policy, "br": args.br}}, fh, indent=1)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
