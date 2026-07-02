"""Fable critique probes (light CPU, headroom-gate class — NOT training).

Probe A — hybrid mechanics: does cross-event double assignment happen, and does it strand a
truck orbiting a zero-demand node (assigned_target never clears once its demand is served by
the other truck)? Instrument greedy episodes (no-attack + scripted route-reach attack).

Probe B — headroom decomposition: how much of the H5 "+79%" is unavoidable detour cost that NO
policy can recover? Run greedy under a PERMANENT blockade of the gateway set (the strongest
static attack the antagonist can approximate by renewing blocks) -> the pure-detour floor.
recoverable ~= attacked_greedy - floor (upper bound on what a clever policy could win back).

    PYTHONPATH=. python scratch/critique_probes.py
"""
from __future__ import annotations

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper, DecisionType
from src.envs.assignment_factory import make_hybrid_assign_env
from src.baselines.greedy_dispatch import hybrid_greedy_policy, no_antagonist_policy, run_episode

GATEWAYS = [("0", "1"), ("0", "129"), ("0", "32"), ("128", "130"), ("163", "164")]


def _cfg(reach: str = "route") -> SMDPConfig:
    return SMDPConfig(
        max_ticks=1500, reward_mode="latency", routing_mode="hybrid", routing_corridor_slack=2.0,
        antagonist_interval=25, congestion_duration=125, congestion_budget=4000.0,
        congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(1.0,),
        max_antag_actions_per_event=1, antag_reach=reach)


def route_reach_attacker(event):
    lbe = event.antagonist_action_mask.get("levels_by_edge", {})
    if not lbe:
        return None
    e = sorted(lbe, key=repr)[0]
    return (e, max(lbe[e]))


def instrumented_episode(antag_policy, label: str):
    """Greedy hybrid episode with assignment/serve tracing."""
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=_cfg("route"))
    policy = hybrid_greedy_policy(smdp)
    event = smdp.reset_decision_env()
    env = smdp.env
    assignments = []   # (tick, truck, node)
    serves = []        # (tick, truck, node)
    wasted_arrivals = []  # truck arrived at assigned target with demand already 0

    while not event.done:
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            # snapshot assignment state before acting
            pre = {tid: t.assigned_target for tid, t in env.trucks.items()}
            actions = policy(event)
            # detect assignment actions: unassigned truck choosing a demand node
            for tid, node in actions.items():
                if pre.get(tid) is None and env.trucks[tid].load > 0:
                    assignments.append((env.time, tid, node))
            next_event, _ = smdp.step_protagonist(actions)
        elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            action = antag_policy(event)
            next_event, _ = smdp.step_antagonist(action)
        else:
            next_event = smdp.advance_until_decision()
        for tick_info in next_event.info.get("events", []):
            for d in tick_info.get("deliveries", []):
                serves.append((tick_info["time"], d["truck_id"], d["node"]))
        event = next_event

    # end-state diagnosis
    stuck = []
    for tid, t in env.trucks.items():
        tgt = t.assigned_target
        if tgt is not None and tgt != t.home_depot and env.graph.nodes[tgt].get("demand", 0.0) <= 0:
            stuck.append((tid, tgt))
    # double assignments: node assigned to >1 truck over the episode while it had 1 unit demand
    from collections import Counter
    per_node = Counter(n for _, _, n in assignments)
    doubles = {n: c for n, c in per_node.items() if c > 1}

    delivered = len(serves)
    last_serve = max((t for t, _, _ in serves), default=None)
    print(f"[{label}] delivered={delivered}/8  ticks={env.time}  last_serve_tick={last_serve}")
    print(f"   assignments={len(assignments)}  multi-assigned nodes={doubles}")
    print(f"   trucks stuck on zero-demand target at end: {stuck}")
    return delivered, doubles, stuck


def permanent_blockade_floor():
    """Greedy vs a PERMANENT blockade of gateway subsets: the pure unavoidable-detour cost."""
    import itertools
    base_smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=_cfg("route"))
    g_no = run_episode(base_smdp, hybrid_greedy_policy(base_smdp), no_antagonist_policy)["total_wait"]
    print(f"[floor] greedy no-attack: {g_no:.0f}")

    for k in (1, 2, 3, 5):
        blocked = GATEWAYS[:k]

        def make_env(blocked=blocked):
            env = make_hybrid_assign_env()
            for e in blocked:
                if env.graph.has_edge(*e):
                    env.set_congestion(e, 1.0)
                    # persist across reset(): bake into the initial graph copy
                    env._initial_graph.edges[e[0], e[1]]["congestion_level"] = 1.0
                    env._initial_graph.edges[e[0], e[1]]["effective_weight"] = (
                        env._initial_graph.edges[e[0], e[1]]["distance"] / 1e-6)
            return env

        smdp = SMDPDecisionWrapper(env_factory=make_env, config=_cfg("route"))
        r = run_episode(smdp, hybrid_greedy_policy(smdp), no_antagonist_policy)
        print(f"[floor] greedy w/ {k} gateway(s) PERMANENTLY blocked: total_wait={r['total_wait']:.0f} "
              f"(+{100*(r['total_wait']-g_no)/g_no:.1f}%)  delivered={r['delivered']}/8")


def budget_sweep():
    """Post-fix budget re-tune: scripted route-reach attack cost on greedy as a function of the
    antagonist budget. Target band for the robustness matrix: hurts (~+30-50%) without saturating."""
    from dataclasses import replace
    base = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=_cfg("route"))
    g_no = run_episode(base, hybrid_greedy_policy(base), no_antagonist_policy)["total_wait"]
    print(f"[sweep] greedy no-attack: {g_no:.0f}")
    for budget in (250, 500, 1000, 1500, 2000, 3000, 4000):
        cfg = replace(_cfg("route"), congestion_budget=float(budget))
        smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=cfg)
        r = run_episode(smdp, hybrid_greedy_policy(smdp), route_reach_attacker)
        print(f"[sweep] budget {budget:5d}: total_wait={r['total_wait']:6.0f} "
              f"(+{100*(r['total_wait']-g_no)/g_no:5.1f}%)  delivered={r['delivered']}/8  "
              f"blocks={r['budget_used']/125:.0f}  end_tick={r['ticks']}")


def main():
    print("=== Probe A: hybrid mechanics (double assignment / orbit deadlock) ===")
    instrumented_episode(no_antagonist_policy, "greedy no-attack")
    instrumented_episode(route_reach_attacker, "greedy vs route-reach")
    print()
    print("=== Probe B: unavoidable-detour floor vs the +79% attack ===")
    permanent_blockade_floor()
    print()
    print("=== Probe C: post-fix antagonist budget sweep (scripted route-reach vs greedy) ===")
    budget_sweep()


if __name__ == "__main__":
    main()
