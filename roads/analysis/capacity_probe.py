#!/usr/bin/env python3
"""Capacity lever probe: greedy rollouts only, no training.

Measures whether raising truck capacity from 1 to {3, 5} strengthens the exploitability gap
between a deterministic and an eps-randomised-assignment greedy dispatcher under a targeted
route-reach attacker, on paired demand instances in the contested arena.

Run: PYTHONPATH=. .venv/bin/python analysis/capacity_probe.py
"""

from __future__ import annotations

import random
import statistics

from src.baselines.attackers import targeted_block_policy
from src.baselines.greedy_dispatch import (
    _congestion_aware_distance, _id_key, greedy_insertion_policy,
    no_antagonist_policy, run_episode)
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.contested import contested_config, make_contested_env

CAPACITIES = [1.0, 3.0, 5.0]
N_INSTANCES = 12
N_ROLLOUTS_RAND = 3          # greedy_rand is stochastic -> average
EPS = 0.3                    # assignment-randomisation rate for the lever probe
SEED_BASE = 10_000_019
RATE = 0.06


def eps_random_greedy(smdp, rng):
    """Greedy insertion, but with prob EPS a free truck takes a RANDOM valid request instead of the
    nearest (assignment unpredictability). Sequential claiming preserved."""
    def policy(event):
        env = smdp.env
        actions, claimed = {}, set()
        for truck_id in sorted(event.protagonist_action_mask):
            dests = [d for d in event.protagonist_action_mask[truck_id] if d not in claimed]
            if not dests:
                continue
            requests = [d for d in dests if env.graph.nodes[d]["demand"] > 0.0]
            source = env.trucks[truck_id].current_node
            if requests and source is not None:
                if rng.random() < EPS:
                    best = rng.choice(sorted(requests, key=_id_key))
                else:
                    best = min(requests, key=lambda d: (_congestion_aware_distance(env, source, d), _id_key(d)))
                actions[truck_id] = best
                claimed.add(best)
            else:
                actions[truck_id] = dests[0]
        return actions
    return policy


def _mk(cap, seed):
    return SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=RATE, demand_seed=seed, truck_capacity=cap),
        config=contested_config())


def _greedy(cap, seed, attacked):
    smdp = _mk(cap, seed)
    atk = targeted_block_policy(smdp) if attacked else no_antagonist_policy
    r = run_episode(smdp, greedy_insertion_policy(smdp), atk)
    return r["total_wait"], r["delivery_rate"]


def _greedy_rand(cap, seed, attacked, rollout):
    smdp = _mk(cap, seed)
    rng = random.Random((seed << 8) ^ (rollout << 2) ^ int(attacked))
    atk = targeted_block_policy(smdp) if attacked else no_antagonist_policy
    return run_episode(smdp, eps_random_greedy(smdp, rng), atk)["total_wait"]


def main():
    print(f"Capacity lever probe: caps={CAPACITIES}, {N_INSTANCES} instances, eps={EPS}, "
          f"{N_ROLLOUTS_RAND} rollouts for randomised\n")
    seeds = [SEED_BASE + i for i in range(N_INSTANCES)]
    header = f"{'cap':>4} | {'W_clean':>8} {'deliv':>6} | {'D_det(bite)':>11} | {'D_rand':>8} | {'LEVER':>7} | {'clean_cost':>10}"
    print(header); print("-" * len(header))
    rows = []
    for cap in CAPACITIES:
        wc, dc, ddet, drand, ccost = [], [], [], [], []
        for s in seeds:
            w_clean, deliv = _greedy(cap, s, attacked=False)
            w_det_atk, _ = _greedy(cap, s, attacked=True)
            wc.append(w_clean); dc.append(deliv)
            ddet.append(w_det_atk - w_clean)
            # randomised greedy: clean + attacked, averaged over rollouts
            wr_clean = statistics.mean(_greedy_rand(cap, s, False, r) for r in range(N_ROLLOUTS_RAND))
            wr_atk = statistics.mean(_greedy_rand(cap, s, True, r) for r in range(N_ROLLOUTS_RAND))
            drand.append(wr_atk - wr_clean)
            ccost.append(wr_clean - w_clean)
        row = dict(cap=cap, W_clean=statistics.mean(wc), deliv=statistics.mean(dc),
                   D_det=statistics.mean(ddet), D_rand=statistics.mean(drand),
                   lever=statistics.mean(ddet) - statistics.mean(drand),
                   clean_cost=statistics.mean(ccost))
        rows.append(row)
        print(f"{cap:>4.0f} | {row['W_clean']:>8.0f} {row['deliv']:>6.2f} | {row['D_det']:>11.0f} | "
              f"{row['D_rand']:>8.0f} | {row['lever']:>7.0f} | {row['clean_cost']:>10.0f}")

    print("\nReading:")
    print("  - competence: greedy delivery rate should stay healthy (multi-stop working, no bug).")
    print("  - attack bite: D_det should stay clearly > 0 (route-reach attacker still hurts).")
    print("  - LEVER (D_det - D_rand): higher = unpredictability reduces attacker damage MORE =")
    print("    more room for adversarial training to pay. Compare across capacities vs clean_cost.")
    if len(rows) > 1:
        base = rows[0]
        for r in rows[1:]:
            dl = r['lever'] - base['lever']
            print(f"  - cap {r['cap']:.0f} vs 1: lever {dl:+.0f} ({'STRONGER' if dl>0 else 'weaker'}), "
                  f"clean_cost {r['clean_cost']-base['clean_cost']:+.0f}")


if __name__ == "__main__":
    main()
