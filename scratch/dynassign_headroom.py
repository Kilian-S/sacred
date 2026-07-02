"""Headroom gate for the Stage-1.5 dynamic assignment rung (run BEFORE spending training CPU).

Question: in the dynamic (Poisson) regime, does a better dispatcher beat nearest-greedy — both
statically and under attack — and at what load? If even a clairvoyant can't beat greedy under
attack, destination-mode assignment is too thin (pull next-hop routing forward); if it can, train.

Compares total_wait (telescoped latency, lower=better) over N fixed Poisson demand seeds, swept
over arrival rate lambda, for:
  * greedy-insertion (nearest, congestion-aware)         -- the baseline to beat
  * urgency (oldest-first, congestion-aware)             -- a simple non-myopic heuristic (uses age)
  * clairvoyant (perfect-foresight free-flow 2-truck scheduler) -- the absolute ceiling (no-attack)
under {no-attack, fixed heuristic adversary}. Reports mean+/-std + a load proxy (delivery rate /
residual queue at the horizon) to locate rho~1, and a headroom verdict.

  PYTHONPATH=. python scratch/dynassign_headroom.py                  # full sweep
  PYTHONPATH=. python scratch/dynassign_headroom.py --rates 0.04 --seeds 3   # quick check
"""

from __future__ import annotations

import argparse
import statistics

import networkx as nx

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_dynamic_assign_env
from src.baselines.greedy_dispatch import (
    greedy_insertion_policy, urgency_dispatch_policy, no_antagonist_policy, run_episode,
)


def _cfg(max_ticks: int, budget: float = 4000.0) -> SMDPConfig:
    """Matches the dynassign branch in scripts/train_sacred.py (budget overridable for the gate)."""
    return SMDPConfig(
        max_ticks=max_ticks, antagonist_interval=25, congestion_duration=120,
        congestion_budget=budget, congestion_cooldown=0, congestion_cost=0.1,
        reward_mode="latency", routing_mode="destination", congestion_levels=(1.0,),
        max_antag_actions_per_event=1,
    )


def congest_near_trucks_antagonist(smdp: SMDPDecisionWrapper):
    """Fixed heuristic adversary: each decision, congest an allowed edge (near a truck) at max
    level — deterministic, spends the budget. Stand-in for a trained antagonist at gate time."""
    def pol(event):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        edge = sorted(lbe.keys(), key=repr)[0]
        return (edge, max(lbe[edge]))
    return pol


def clairvoyant_total_wait(env, max_ticks: int) -> float:
    """Perfect-foresight, free-flow ceiling: schedule every (known) request on the 2 depot-trucks
    to minimise total latency, respecting truck busy-time (capacity-1 round trips) and the 2-depot
    geometry. Free-flow (no congestion) + full foresight => a strong reference. List scheduling
    (assign each request, in arrival order, to the depot-truck delivering it soonest).

    Wait is **truncated at the horizon** (a request outstanding from arrival to min(delivery, T)
    contributes min(delivery,T)-arrival) to match run_episode's telescoped total_wait — otherwise
    the clairvoyant counts post-horizon waits the env never accrues and looks worse under overload."""
    depots = list(getattr(env, "assignment_depots", []))
    speed = env.truck_speed
    dist_from = {d: nx.single_source_dijkstra_path_length(env.graph, d, weight="distance") for d in depots}

    requests = []
    for tick, node, size in sorted(env._arrival_schedule, key=lambda r: r[0]):
        requests.extend([(tick, node)] * int(round(size)))

    free_at = {d: 0.0 for d in depots}  # each truck identified by its home depot
    total = 0.0
    for arrival, node in requests:
        best = None  # (delivery, depot, travel)
        for d in depots:
            one_way = dist_from[d].get(node)
            if one_way is None:
                continue
            travel = one_way / speed
            delivery = max(free_at[d] + travel, arrival)
            if best is None or delivery < best[0]:
                best = (delivery, d, travel)
        if best is None:
            continue
        delivery, d, travel = best
        total += max(0.0, min(delivery, max_ticks) - arrival)  # truncate at horizon (telescoped)
        free_at[d] = delivery + travel  # return to depot before the next trip
    return total


def _schedule_matched_env(seed: int, rate: float, max_ticks: int):
    """An env whose arrival schedule matches what run_episode sees (factory __init__ draws once,
    reset_decision_env draws again -> replicate the second draw)."""
    env = make_dynamic_assign_env(arrival_rate=rate, demand_seed=seed, max_time=max_ticks)
    env.reset()
    return env


def _ms(xs):
    return statistics.mean(xs), (statistics.pstdev(xs) if len(xs) > 1 else 0.0)


def gate(rates, seeds, max_ticks, budget=400.0):
    cfg = _cfg(max_ticks, budget)
    print(f"=== dynamic-assignment headroom gate (T={max_ticks}, seeds={len(seeds)}) ===")
    print("total_wait mean+/-std (lower=better). gap%>0 = beats greedy.\n")
    header = (f"{'lambda':>7} | {'drate':>5} {'queue':>5} | {'greedy_no':>11} {'greedy_at':>11} | "
              f"{'urg_no%':>7} {'urg_at%':>7} | {'clair':>7} {'clair_gap%':>10}")
    print(header)
    print("-" * len(header))

    for rate in rates:
        g_no, g_at, u_no, u_at, clair, drate, queue = ([] for _ in range(7))
        for k in seeds:
            def mk(seed=k, rate=rate):
                return make_dynamic_assign_env(arrival_rate=rate, demand_seed=seed, max_time=max_ticks)
            s = SMDPDecisionWrapper(env_factory=mk, config=cfg)
            r = run_episode(s, greedy_insertion_policy(s), no_antagonist_policy)
            g_no.append(r["total_wait"]); drate.append(r["delivery_rate"]); queue.append(r["num_requests"] - r["delivered"])
            s = SMDPDecisionWrapper(env_factory=mk, config=cfg)
            g_at.append(run_episode(s, greedy_insertion_policy(s), congest_near_trucks_antagonist(s))["total_wait"])
            s = SMDPDecisionWrapper(env_factory=mk, config=cfg)
            u_no.append(run_episode(s, urgency_dispatch_policy(s), no_antagonist_policy)["total_wait"])
            s = SMDPDecisionWrapper(env_factory=mk, config=cfg)
            u_at.append(run_episode(s, urgency_dispatch_policy(s), congest_near_trucks_antagonist(s))["total_wait"])
            clair.append(clairvoyant_total_wait(_schedule_matched_env(k, rate, max_ticks), max_ticks))

        gno, gat = _ms(g_no)[0], _ms(g_at)[0]
        uno, uat = _ms(u_no)[0], _ms(u_at)[0]
        cl = _ms(clair)[0]
        # gap% > 0 means the alternative beats greedy (lower total_wait).
        urg_no_pct = 100 * (gno - uno) / gno if gno else 0.0
        urg_at_pct = 100 * (gat - uat) / gat if gat else 0.0
        clair_gap_pct = 100 * (gat - cl) / gat if gat else 0.0  # max recoverable under attack vs ideal
        print(f"{rate:>7.3f} | {_ms(drate)[0]:>5.2f} {_ms(queue)[0]:>5.0f} | "
              f"{gno:>5.0f}+/-{_ms(g_no)[1]:<4.0f} {gat:>5.0f}+/-{_ms(g_at)[1]:<4.0f} | "
              f"{urg_no_pct:>6.1f}% {urg_at_pct:>6.1f}% | {cl:>7.0f} {clair_gap_pct:>9.1f}%")

    print("\nReading: pick lambda where drate is loaded but not collapsed (~0.7-0.97, queue>0) AND")
    print("there is headroom under attack (urg_at% > 0, and/or clair_gap% large = room above greedy).")
    print("If clair_gap% under attack is small everywhere -> destination-mode is too thin; pull")
    print("next-hop routing forward before training.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rates", type=str, default="0.025,0.04,0.06,0.08")
    p.add_argument("--seeds", type=int, default=8)
    p.add_argument("--max-ticks", type=int, default=800)
    p.add_argument("--budget", type=float, default=400.0)
    args = p.parse_args()
    print(f"[congestion_budget={args.budget}]")
    gate([float(x) for x in args.rates.split(",")], list(range(args.seeds)), args.max_ticks, args.budget)


if __name__ == "__main__":
    main()
