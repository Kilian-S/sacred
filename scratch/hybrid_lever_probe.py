#!/usr/bin/env python3
"""Routing-unpredictability lever probe on the HYBRID arena (gen07 arena scoping, 2026-07-06).

The powered destination-mode sweep (scratch/stress_sweep.py) found the ASSIGNMENT-unpredictability
lever thin at every load (~2-7% of D). But destination mode AUTO-ROUTES (Dijkstra), so it
structurally cannot express ROUTING unpredictability, which is the thesis's actual mechanism
("unpredictable roads deny the ambush"). This probe measures the routing lever on the hybrid arena
(routing_mode="hybrid", route-reach attacker), where the policy chooses each edge:

  D_det   = W(hybrid_greedy_det, targeted) - W(hybrid_greedy_det, none)   [deterministic routing]
  D_rand  = W(hybrid_greedy_eps_routing, targeted) - W(..., none)         [eps-random forward edge]
  LEVER   = D_det - mean(D_rand)   (higher = routing unpredictability reduces the route-reach
            attacker's damage more; the room a learned routing policy could exploit)

Static demand -> the deterministic greedy is a single trajectory; the CI comes from the eps-rollout
distribution (the relevant uncertainty). Compared against the ~2-7% destination assignment lever,
this says whether the exploitability lever really lives in routing control (=> gen07 wants a
routing arena) or is thin everywhere (=> the benefit is structurally small in this problem).

Run: PYTHONPATH=. .venv/bin/python scratch/hybrid_lever_probe.py
"""

from __future__ import annotations

import math
import random
import statistics

import networkx as nx

from scripts.evaluate_hybrid import hybrid_config
from src.baselines.attackers import targeted_block_policy
from src.baselines.greedy_dispatch import (
    _congestion_aware_distance, _id_key, hybrid_greedy_policy,
    no_antagonist_policy, run_episode)
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.assignment_factory import make_hybrid_assign_env

N_ROLLOUTS = 40
EPS_VALUES = [0.2, 0.4]


def eps_routing_greedy(smdp, rng, eps):
    """Hybrid greedy, but on the ROUTING branch pick a random forward option with prob eps (routing
    unpredictability). Assignment branch stays greedy (isolates the routing lever)."""
    def policy(event):
        env = smdp.env
        actions, claimed = {}, set()
        for truck_id in sorted(event.protagonist_action_mask):
            options = event.protagonist_action_mask[truck_id]
            if not options:
                continue
            truck = env.trucks[truck_id]
            source = truck.current_node
            if source is None:
                continue
            if truck.assigned_target is None:
                avail = [n for n in options if n not in claimed]
                if not avail:
                    continue
                best = min(avail, key=lambda d: (_congestion_aware_distance(env, source, d), _id_key(d)))
                actions[truck_id] = best
                claimed.add(best)
            else:
                if rng.random() < eps and len(options) > 1:
                    actions[truck_id] = rng.choice(sorted(options, key=_id_key))
                else:
                    try:
                        path = nx.dijkstra_path(env.graph, source, truck.assigned_target, weight="effective_weight")
                        nxt = path[1] if len(path) > 1 else options[0]
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        nxt = options[0]
                    actions[truck_id] = nxt if nxt in options else options[0]
        return actions
    return policy


def _mk():
    return SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=hybrid_config())


def _det(attacked):
    smdp = _mk()
    atk = targeted_block_policy(smdp) if attacked else no_antagonist_policy
    return run_episode(smdp, hybrid_greedy_policy(smdp), atk)["total_wait"]


def _rand(attacked, eps, rollout):
    smdp = _mk()
    rng = random.Random((rollout << 3) ^ int(attacked) ^ int(eps * 100))
    atk = targeted_block_policy(smdp) if attacked else no_antagonist_policy
    return run_episode(smdp, eps_routing_greedy(smdp, rng, eps), atk)["total_wait"]


def _ci95(xs):
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0


def main():
    w_clean_det = _det(attacked=False)
    w_atk_det = _det(attacked=True)
    d_det = w_atk_det - w_clean_det
    print(f"HYBRID routing-lever probe ({N_ROLLOUTS} eps-rollouts):\n")
    print(f"deterministic hybrid greedy: W_clean={w_clean_det:.0f}  W_targeted={w_atk_det:.0f}  "
          f"D_det={d_det:.0f} ({100*d_det/max(1,w_clean_det):.0f}% of clean)\n")
    header = f"{'eps':>5} | {'D_rand (95% CI)':>20} | {'LEVER':>7} | {'clean_cost':>10} | {'lever %D_det':>12}"
    print(header); print("-" * len(header))
    for eps in EPS_VALUES:
        rand_clean = [_rand(False, eps, r) for r in range(N_ROLLOUTS)]
        rand_atk = [_rand(True, eps, r) for r in range(N_ROLLOUTS)]
        d_rand = [a - statistics.mean(rand_clean) for a in rand_atk]  # vs mean clean
        d_rand_m, d_rand_ci = statistics.mean(d_rand), _ci95(d_rand)
        lever = d_det - d_rand_m
        clean_cost = statistics.mean(rand_clean) - w_clean_det
        pct = 100 * lever / max(1, d_det)
        print(f"{eps:>5.1f} | {d_rand_m:>8.0f} +/- {d_rand_ci:>7.0f} | {lever:>7.0f} | "
              f"{clean_cost:>10.0f} | {pct:>11.0f}%")
    print("\nCompare LEVER %D_det here (routing) vs the ~2-7% destination assignment lever. If routing "
          "is materially larger, the exploitability lever lives in ROUTING control -> gen07 wants a "
          "routing arena (dynamic-hybrid); if similarly thin, the benefit is structurally small.")


if __name__ == "__main__":
    main()
