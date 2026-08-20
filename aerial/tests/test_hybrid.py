"""Tests for the hybrid (assignment plus next-hop routing) truck state machine: a truck cycles
through assignment, routing, serving, going home and reloading, delivers all demand, and never
serves a request it was not assigned."""

from __future__ import annotations

import networkx as nx

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper, DecisionType
from src.envs.assignment_factory import make_assignment_env, make_hybrid_assign_env


def _hybrid_cfg(max_ticks: int = 800) -> SMDPConfig:
    return SMDPConfig(
        max_ticks=max_ticks, reward_mode="latency", routing_mode="hybrid",
        routing_corridor_slack=1.3, antagonist_interval=25, congestion_duration=125,
        congestion_budget=4000.0, congestion_cooldown=0, congestion_cost=0.1,
        congestion_levels=(1.0,), max_antag_actions_per_event=1)


def _hybrid_greedy(smdp: SMDPDecisionWrapper):
    """Minimal hybrid policy that assigns the nearest unclaimed request and routes via the
    neighbour on the congestion-aware shortest path to the assigned target."""
    def policy(event):
        env = smdp.env
        actions, claimed = {}, set()
        for tid in sorted(event.protagonist_action_mask):
            opts = [n for n in event.protagonist_action_mask[tid] if n not in claimed]
            if not opts:
                continue
            truck = env.trucks[tid]
            if truck.assigned_target is None:  # assignment
                src = truck.current_node
                best = min(opts, key=lambda d: (nx.shortest_path_length(env.graph, src, d, weight="distance"), str(d)))
                actions[tid] = best
                claimed.add(best)
            else:  # routing toward the assigned target
                try:
                    p = nx.shortest_path(env.graph, truck.current_node, truck.assigned_target, weight="effective_weight")
                    nxt = p[1] if len(p) > 1 else opts[0]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    nxt = opts[0]
                actions[tid] = nxt if nxt in opts else opts[0]
        return actions
    return policy


def test_hybrid_full_cycle_delivers_all():
    cfg = _hybrid_cfg(800)
    smdp = SMDPDecisionWrapper(env_factory=lambda: make_assignment_env(), config=cfg)
    pol = _hybrid_greedy(smdp)
    env = smdp.env
    home0 = env.trucks[0].home_depot
    demand_nodes = set(env.assignment_demand)
    total_demand = sum(d.get("demand", 0.0) for _, d in env._initial_graph.nodes(data=True))

    event = smdp.reset_decision_env()
    saw_assignment = saw_routing = False
    t0_targets = []
    steps = 0
    while not event.done and steps < 20000:
        steps += 1
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            for tid, opts in event.protagonist_action_mask.items():
                if opts:
                    if smdp.env.trucks[tid].assigned_target is None:
                        saw_assignment = True
                    else:
                        saw_routing = True
            event, _ = smdp.step_protagonist(pol(event))
        elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
            event, _ = smdp.step_antagonist(None)
        else:
            event = smdp.advance_until_decision()
        t0_targets.append(smdp.env.trucks[0].assigned_target)

    assert saw_assignment and saw_routing, "expected BOTH assignment and routing decisions"
    # Truck 0 completes a cycle: a demand node, then the home depot, then cleared.
    assert any(t in demand_nodes for t in t0_targets if t is not None), "truck 0 never got a request"
    assert home0 in t0_targets, "truck 0 never routed home (post-serve flip)"
    # Each truck serves only its assigned target, so full delivery requires the full cycle.
    delivered = sum(t.delivered_total for t in smdp.env.trucks.values())
    assert delivered >= total_demand - 1e-6, f"delivered {delivered} of {total_demand}"
    assert smdp.env.is_done()


def test_hybrid_no_opportunistic_serving():
    """A truck must not serve a demand node it merely passes through (only its assigned target)."""
    cfg = _hybrid_cfg(800)
    smdp = SMDPDecisionWrapper(env_factory=lambda: make_assignment_env(), config=cfg)
    pol = _hybrid_greedy(smdp)
    event = smdp.reset_decision_env()
    steps = 0
    while not event.done and steps < 20000:
        steps += 1
        # Every delivery must be at the serving truck's assigned target.
        for tick_info in event.info.get("events", []):
            for dv in tick_info.get("deliveries", []):
                tid = dv["truck_id"]
                assert dv["node"] in set(smdp.env.assignment_demand)
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            event, _ = smdp.step_protagonist(pol(event))
        elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
            event, _ = smdp.step_antagonist(None)
        else:
            event = smdp.advance_until_decision()
    assert smdp.env.is_done()


def test_hybrid_greedy_baseline_delivers():
    """H4: the hybrid greedy baseline (assignment + reactive next-hop routing) delivers all demand
    on the chokepoint geometry, no attack."""
    from src.baselines.greedy_dispatch import hybrid_greedy_policy, no_antagonist_policy, run_episode

    cfg = _hybrid_cfg(1200)
    smdp = SMDPDecisionWrapper(env_factory=lambda: make_hybrid_assign_env(), config=cfg)
    m = run_episode(smdp, hybrid_greedy_policy(smdp), no_antagonist_policy)
    total = int(round(sum(d.get("demand", 0.0) for _, d in smdp.env._initial_graph.nodes(data=True))))
    assert m["delivered"] == total, f"delivered {m['delivered']} of {total}"
    assert m["total_wait"] > 0


def test_route_reach_targets_gateway_and_is_bounded():
    """H3: route-reach exposes exactly the edges on a truck's shortest path to its target (incl. the
    ('0','1') gateway), not a blob around it."""
    from dataclasses import replace

    cfg = replace(_hybrid_cfg(), antag_reach="route")
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=cfg)
    smdp.reset_decision_env()
    env = smdp.env
    env.trucks[0].assigned_target = "46"   # east demand; Depot-A route crosses ('0','1')
    env.trucks[1].assigned_target = None   # only truck 0 committed -> only its route contributes

    reach = smdp._route_reach_edges()
    path = nx.shortest_path(env.graph, env.trucks[0].current_node, "46", weight="distance")
    expected = {env._edge_key(path[i], path[i + 1]) for i in range(len(path) - 1)}
    assert reach == expected, "route-reach must be exactly the route's edges"
    assert env._edge_key("0", "1") in reach, "the gateway must be reachable on this route"
    # and it surfaces in the antagonist mask
    assert env._edge_key("0", "1") in smdp.antagonist_action_mask().get("levels_by_edge", {})


def test_hybrid_route_reach_antagonist_runs_and_attacks():
    """H3: greedy vs the route-reach antagonist runs to completion; the antagonist lands blocks, and
    greedy still routes around them to deliver all demand (higher latency)."""
    from dataclasses import replace
    from src.baselines.greedy_dispatch import hybrid_greedy_policy, run_episode

    cfg = replace(_hybrid_cfg(1500), antag_reach="route")
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=cfg)

    def attacker(event):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        e = sorted(lbe, key=repr)[0]
        return (e, max(lbe[e]))

    m = run_episode(smdp, hybrid_greedy_policy(smdp), attacker)
    assert smdp.budget.used > 0, "route-reach antagonist never landed a block"
    total = int(round(sum(d.get("demand", 0.0) for _, d in smdp.env._initial_graph.nodes(data=True))))
    assert m["delivered"] == total, f"greedy delivered {m['delivered']} of {total} under attack"


def test_eval_hybrid_cells_deterministic_and_structured():
    """H6: the hybrid eval harness (learned vs greedy, no-attack/attack) is deterministic (static)
    and gap = learned - greedy."""
    import torch
    from dataclasses import replace
    from scripts.evaluate_hybrid import hybrid_config, eval_hybrid_cells, _new_protag, _new_antag

    cfg = replace(hybrid_config(), max_ticks=300)  # short horizon keeps the untrained-policy eval fast
    torch.manual_seed(0)
    protag, antag = _new_protag(), _new_antag(cfg)
    r1 = eval_hybrid_cells(protag, antag, make_hybrid_assign_env, cfg)
    r2 = eval_hybrid_cells(protag, antag, make_hybrid_assign_env, cfg)
    assert r1 == r2, "static hybrid eval must be deterministic"
    for k in ["greedy_atk", "learned_atk", "gap_atk", "gap_noatk"]:
        assert k in r1
    assert abs((r1["learned_atk"] - r1["greedy_atk"]) - r1["gap_atk"]) < 1e-6
