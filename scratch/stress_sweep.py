#!/usr/bin/env python3
"""Powered stress (load) sweep at capacity 1 (gen07 arena scoping, 2026-07-06).

Follow-up to scratch/capacity_probe.py, which showed the exploitability lever is a STRESS
phenomenon (capacity slack destroys it) and hinted at a load sweet spot near lambda=0.08 off only
12 instances. This pins it with power: per-instance lever with a 95% CI, so we can tell the real
signal from the difference-of-differences noise, and find the load that maximises the lever while
keeping greedy trainable (delivery above the collapse band).

Metrics per lambda (capacity 1, contested arena = dynassign + route reach), paired demand seeds:
  * W_clean, delivery      : deterministic greedy, no attack (competence / trainability).
  * D_det                  : W(greedy_det, targeted) - W(greedy_det, none), the attacker's bite.
  * LEVER (mean +/- 95%CI) : per instance, D_det_i - D_rand_i, where D_rand_i uses eps-randomised
                             greedy averaged over rollouts. > 0 (CI excluding 0) = unpredictability
                             measurably reduces attacker damage; the room adversarial training has.
  * clean_cost             : W(greedy_rand, none) - W(greedy_det, none), the price of that mixing.
  * ratio                  : lever / clean_cost (> 1 = unpredictability pays for itself).

Run: PYTHONPATH=. .venv/bin/python scratch/stress_sweep.py
"""

from __future__ import annotations

import math
import random
import statistics

from src.baselines.attackers import targeted_block_policy
from src.baselines.greedy_dispatch import greedy_insertion_policy, no_antagonist_policy, run_episode
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.contested import contested_config, make_contested_env
from scratch.capacity_probe import eps_random_greedy

CAP = 1.0
LAMBDAS = [0.06, 0.07, 0.08, 0.09, 0.10]
N_INSTANCES = 40
N_ROLLOUTS = 5
EPS = 0.3
SEED_BASE = 10_000_019


def _mk(rate, seed):
    return SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=rate, demand_seed=seed, truck_capacity=CAP),
        config=contested_config())


def _det(rate, seed, attacked):
    smdp = _mk(rate, seed)
    atk = targeted_block_policy(smdp) if attacked else no_antagonist_policy
    r = run_episode(smdp, greedy_insertion_policy(smdp), atk)
    return r["total_wait"], r["delivery_rate"]


def _rand(rate, seed, attacked, rollout):
    smdp = _mk(rate, seed)
    rng = random.Random((seed << 8) ^ (rollout << 2) ^ int(attacked))
    atk = targeted_block_policy(smdp) if attacked else no_antagonist_policy
    return run_episode(smdp, eps_random_greedy(smdp, rng), atk)["total_wait"]


def _ci95(xs):
    if len(xs) < 2:
        return 0.0
    return 1.96 * statistics.stdev(xs) / math.sqrt(len(xs))


def main():
    print(f"Powered stress sweep: cap={CAP:.0f}, lambdas={LAMBDAS}, {N_INSTANCES} instances, "
          f"{N_ROLLOUTS} rollouts, eps={EPS}\n", flush=True)
    header = (f"{'lambda':>7} | {'W_clean':>8} {'deliv':>6} | {'D_det':>7} | "
              f"{'LEVER (95% CI)':>20} | {'clean_cost':>10} | {'ratio':>6} | sig?")
    print(header, flush=True)
    print("-" * len(header), flush=True)
    for rate in LAMBDAS:
        wcs, dcs, ddets, levers, ccosts = [], [], [], [], []
        for i in range(N_INSTANCES):
            seed = SEED_BASE + i
            w0, deliv = _det(rate, seed, False)
            wa, _ = _det(rate, seed, True)
            d_det = wa - w0
            rand_clean = statistics.mean(_rand(rate, seed, False, r) for r in range(N_ROLLOUTS))
            rand_atk = statistics.mean(_rand(rate, seed, True, r) for r in range(N_ROLLOUTS))
            d_rand = rand_atk - rand_clean
            wcs.append(w0); dcs.append(deliv); ddets.append(d_det)
            levers.append(d_det - d_rand); ccosts.append(rand_clean - w0)
        lever_m, lever_ci = statistics.mean(levers), _ci95(levers)
        cost_m = statistics.mean(ccosts)
        ratio = lever_m / cost_m if abs(cost_m) > 1 else float("nan")
        sig = "YES" if abs(lever_m) > lever_ci and lever_m > 0 else "no"
        print(f"{rate:>7.2f} | {statistics.mean(wcs):>8.0f} {statistics.mean(dcs):>6.2f} | "
              f"{statistics.mean(ddets):>7.0f} | {lever_m:>8.0f} +/- {lever_ci:>7.0f} | "
              f"{cost_m:>10.0f} | {ratio:>6.2f} | {sig}", flush=True)

    print("\nDecision inputs: pick the lambda with the largest CI-positive lever whose delivery is "
          "still trainable (well above the gen06 collapse band ~0.2-0.3). That load, at capacity 1, "
          "is the gen07 arena's operating point.", flush=True)


if __name__ == "__main__":
    main()
