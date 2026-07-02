"""H5 — light headroom check for the Stage-2 hybrid rung (NOT a stop-gate).

Under the route-reach antagonist, does an *anticipatory* policy beat *reactive* greedy? Static demand
is deterministic, so each cell is a single episode (no seed noise). Reports:
  * greedy vs no-attack        -> the ceiling (best case, no adversary)
  * greedy vs route-reach      -> reactive baseline under attack
  * anticipatory vs route-reach-> a policy that PRE-EMPTIVELY avoids the known gateways
  * greedy vs leashed          -> sanity that route-reach is a stronger adversary than leashed
Headroom exists if anticipatory beats reactive-greedy under attack (a learned policy could too).

    PYTHONPATH=. python scratch/hybrid_headroom.py
"""

from __future__ import annotations

import networkx as nx

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_hybrid_assign_env
from src.baselines.greedy_dispatch import hybrid_greedy_policy, no_antagonist_policy, run_episode, _id_key

# The chokepoints the route-reach adversary targets (from find_chokepoints.py / the geometry search).
GATEWAYS = {("0", "1"), ("0", "129"), ("0", "32"), ("128", "130"), ("163", "164")}


def _cfg(reach: str) -> SMDPConfig:
    return SMDPConfig(
        max_ticks=1500, reward_mode="latency", routing_mode="hybrid", routing_corridor_slack=2.0,
        antagonist_interval=25, congestion_duration=125, congestion_budget=4000.0, congestion_cooldown=0,
        congestion_cost=0.1, congestion_levels=(1.0,), max_antag_actions_per_event=1, antag_reach=reach)


def route_reach_attacker(event):
    lbe = event.antagonist_action_mask.get("levels_by_edge", {})
    if not lbe:
        return None
    e = sorted(lbe, key=repr)[0]
    return (e, max(lbe[e]))


def anticipatory_policy(smdp, penalty: float = 8.0):
    """Hybrid greedy that PRE-EMPTIVELY routes/assigns AROUND the known gateways (treats them as
    costly) — a stand-in for a learned anticipatory policy. If this beats reactive greedy under the
    route-reach adversary, there is headroom for RL to exploit."""
    env = smdp.env
    ek = env._edge_key

    def w(u, v, data):
        base = data["distance"] / max(1e-6, 1.0 - data.get("congestion_level", 0.0))
        return base * (penalty if ek(u, v) in GATEWAYS else 1.0)

    def policy(event):
        actions, claimed = {}, set()
        for tid in sorted(event.protagonist_action_mask):
            opts = event.protagonist_action_mask[tid]
            if not opts:
                continue
            truck = env.trucks[tid]
            src = truck.current_node
            if truck.assigned_target is None:  # assignment: nearest by gateway-avoiding distance
                avail = [n for n in opts if n not in claimed]
                if not avail:
                    continue
                best = min(avail, key=lambda d: (nx.shortest_path_length(env.graph, src, d, weight=w), _id_key(d)))
                actions[tid] = best
                claimed.add(best)
            else:  # routing: among the CORRIDOR options, pick the one best reducing gateway-penalised
                # distance to the target (stays in-corridor, prefers avoiding gateways)
                def pen_dist(n):
                    try:
                        return nx.shortest_path_length(env.graph, n, truck.assigned_target, weight=w)
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        return float("inf")
                actions[tid] = min(opts, key=lambda n: (pen_dist(n), _id_key(n)))
        return actions
    return policy


def run(policy_factory, antag, reach):
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=_cfg(reach))
    return run_episode(smdp, policy_factory(smdp), antag)["total_wait"]


def main():
    g_no = run(hybrid_greedy_policy, no_antagonist_policy, "route")
    g_route = run(hybrid_greedy_policy, route_reach_attacker, "route")
    a_route = run(anticipatory_policy, route_reach_attacker, "route")
    g_leash = run(hybrid_greedy_policy, route_reach_attacker, "leashed")

    print("=== hybrid headroom (static, deterministic) — total_wait, lower=better ===")
    print(f"  greedy   vs no-attack        : {g_no:8.0f}   (ceiling)")
    print(f"  greedy   vs LEASHED antag     : {g_leash:8.0f}   (+{100*(g_leash-g_no)/g_no:.1f}%)")
    print(f"  greedy   vs ROUTE-REACH antag : {g_route:8.0f}   (+{100*(g_route-g_no)/g_no:.1f}%)  <- reactive baseline")
    print(f"  anticip. vs ROUTE-REACH antag : {a_route:8.0f}   (+{100*(a_route-g_no)/g_no:.1f}%)")
    print()
    print(f"  attack cost (route-reach)     : {g_route-g_no:+.0f} ({100*(g_route-g_no)/g_no:+.1f}%) = max recoverable")
    print(f"  anticipation gap vs greedy    : {g_route-a_route:+.0f} ({100*(g_route-a_route)/g_route:+.1f}%)  "
          f"{'<- HEADROOM (anticipation beats reaction)' if a_route < g_route else '<- no gain from this heuristic'}")
    print(f"  route-reach vs leashed        : route-reach is {'STRONGER' if g_route > g_leash else 'not stronger'} "
          f"({g_route-g_leash:+.0f})")


if __name__ == "__main__":
    main()
