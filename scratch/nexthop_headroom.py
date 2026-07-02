"""Adversarial headroom gate for next-hop Stage 0 (before any training).

Question: in next-hop mode, if the antagonist targets the FAST route, does a policy that
takes the SAFE route beat the reactive greedy baseline? If yes, there is a learnable robust
strategy and training is justified. If no, stop and report (as we did for capacity>1).

Compares total_wait (sum of unit latencies, lower=better) for:
  greedy(reactive) vs safe-route(fixed)   under   {no attack, congest-fast, adaptive}.
"""

from __future__ import annotations

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.stage0_factory import make_stage0_nexthop_env
from src.baselines.greedy_dispatch import greedy_next_hop_policy, no_antagonist_policy, run_episode

DEPOT, TARGET = "14", "82"
FAST = ["14", "15", "127", "202", "184", "82"]
SAFE = ["14", "11", "9", "10", "52", "82"]


def cfg() -> SMDPConfig:
    return SMDPConfig(max_ticks=400, reward_mode="latency", routing_mode="next_hop",
                      antagonist_interval=20, congestion_duration=30, congestion_budget=300.0,
                      congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(0.25, 0.5, 0.75, 1.0))


def _edge_keys(env, seq):
    return {env._edge_key(seq[i], seq[i + 1]) for i in range(len(seq) - 1)}


def safe_route_policy(smdp):
    """Next-hop policy that always follows the SAFE route (toward target if loaded, else depot)."""
    def policy(event):
        actions = {}
        for tid, neighbors in event.protagonist_action_mask.items():
            truck = smdp.env.trucks[tid]
            cur = truck.current_node
            goal_seq = SAFE if truck.load > 0 else list(reversed(SAFE))
            nxt = None
            if cur in goal_seq:
                i = goal_seq.index(cur)
                if i + 1 < len(goal_seq):
                    nxt = goal_seq[i + 1]
            actions[tid] = nxt if nxt in neighbors else neighbors[0]
        return actions
    return policy


def congest_fast_antagonist(smdp):
    """Fixed adversary: congest any available FAST-route edge at max affordable level."""
    fast_edges = _edge_keys(smdp.env, FAST)
    def policy(event):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        for edge in sorted(fast_edges, key=repr):
            if edge in lbe and lbe[edge]:
                return (edge, max(lbe[edge]))
        return None
    return policy


def adaptive_antagonist(smdp):
    """Strong adversary: congest the edge the truck is currently traversing (or nearest)."""
    def policy(event):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        on_edge = None
        for t in smdp.env.trucks.values():
            if t.edge is not None:
                on_edge = smdp.env._edge_key(*t.edge)
        if on_edge in lbe and lbe[on_edge]:
            return (on_edge, max(lbe[on_edge]))
        edge = sorted(lbe.keys(), key=repr)[0]
        return (edge, max(lbe[edge]))
    return policy


def run(label, protag, antag):
    smdp = SMDPDecisionWrapper(env_factory=lambda: make_stage0_nexthop_env(), config=cfg())
    r = run_episode(smdp, protag(smdp), antag(smdp) if antag else no_antagonist_policy)
    print(f"  {label:32s} total_wait={r['total_wait']:7.0f}  deliv={r['delivered']}/{r['num_requests']} "
          f"ticks={r['ticks']} budget={r['budget_used']:.0f}")
    return r["total_wait"]


if __name__ == "__main__":
    print("=== next-hop adversarial headroom gate ===")
    print("-- no attack --")
    g0 = run("greedy", greedy_next_hop_policy, None)
    s0 = run("safe-route", safe_route_policy, None)
    print("-- antagonist targets FAST route --")
    gf = run("greedy", greedy_next_hop_policy, congest_fast_antagonist)
    sf = run("safe-route", safe_route_policy, congest_fast_antagonist)
    print("-- adaptive antagonist (upper bound) --")
    ga = run("greedy", greedy_next_hop_policy, adaptive_antagonist)
    sa = run("safe-route", safe_route_policy, adaptive_antagonist)
    print("\n--- verdict ---")
    print(f"no-attack: greedy={g0:.0f} safe={s0:.0f} (fast route is genuinely shorter: greedy<=safe? {g0<=s0})")
    print(f"vs congest-fast: greedy={gf:.0f} safe={sf:.0f} -> safe beats greedy by {gf-sf:.0f} ({100*(gf-sf)/gf:.1f}%)")
    print(f"vs adaptive:     greedy={ga:.0f} safe={sa:.0f}")
    if sf < gf * 0.95:
        print("HEADROOM EXISTS: avoiding the attacked route clearly beats reactive greedy -> training justified.")
    else:
        print("NO CLEAR HEADROOM: route avoidance doesn't beat reactive greedy -> stop and report.")
