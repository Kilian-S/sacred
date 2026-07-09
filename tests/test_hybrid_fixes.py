"""Regression tests for the 2026-07-02 hybrid-rung fixes (see CRITIQUE.md §3 / Probe A).

Covers: cross-event assignment claiming (no double assignment across decision events), release of
a stranded truck when its assigned request is served by another truck (the zombie-orbit bug),
assignment clearing on depot arrival at full load, the send-home fallback when no unclaimed
request remains, early episode termination, and the new observability features
(assigned_target / goal-distance field / antagonist commitment view / narrow-checkpoint slicing).
"""

from __future__ import annotations

import torch

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper, DecisionType
from src.envs.assignment_factory import make_hybrid_assign_env
from src.agents.networks import featurize_state, NODE_FEATURE_DIM
from src.agents.sac import ProtagonistSAC, _clip_x, infer_node_in_dim
from src.baselines.greedy_dispatch import hybrid_greedy_policy, no_antagonist_policy


def _cfg(max_ticks: int = 1500) -> SMDPConfig:
    return SMDPConfig(
        max_ticks=max_ticks, reward_mode="latency", routing_mode="hybrid",
        routing_corridor_slack=2.0, antagonist_interval=25, congestion_duration=125,
        congestion_budget=4000.0, congestion_cooldown=0, congestion_cost=0.1,
        congestion_levels=(1.0,), max_antag_actions_per_event=1, antag_reach="route")


def _fresh():
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=_cfg())
    smdp.reset_decision_env()
    return smdp


def test_cross_event_claiming_excludes_taken_requests():
    smdp = _fresh()
    env = smdp.env
    target = sorted(env.assignment_demand)[0]
    env.trucks[0].assigned_target = target
    candidates = smdp._assignment_candidates(env.trucks[1])
    assert target not in candidates, "a request assigned to another truck must not be offered"
    # unclaimed requests are still offered
    assert any(n in set(env.assignment_demand) for n in candidates)


def test_send_home_fallback_when_no_unclaimed_request():
    smdp = _fresh()
    env = smdp.env
    # exhaust all demand
    for n in list(env.assignment_demand):
        env.graph.nodes[n]["demand"] = 0.0
    env.valid_customers_by_comp = {}
    env.remaining_demand = 0.0
    # a loaded, unassigned truck away from home is sent home (so is_done can fire) ...
    truck = env.trucks[0]
    truck.current_node = "0"
    assert smdp._assignment_candidates(truck) == [truck.home_depot]
    # ... and one already at home just idles (no candidates)
    truck.current_node = truck.home_depot
    assert smdp._assignment_candidates(truck) == []


def test_serve_releases_other_truck_assigned_to_same_node():
    smdp = _fresh()
    env = smdp.env
    node = sorted(env.assignment_demand)[0]
    server, stranded = env.trucks[0], env.trucks[1]
    server.assigned_target = node
    stranded.assigned_target = node  # simulated pre-fix cross-event double assignment
    info = {"deliveries": [], "reloads": []}
    env._serve_demand(server, node, info)
    assert env.graph.nodes[node]["demand"] == 0.0
    assert server.assigned_target == server.home_depot, "server heads home after serving"
    assert stranded.assigned_target is None, "the doubly-assigned truck must be released"


def test_depot_arrival_clears_assignment_even_at_full_load():
    smdp = _fresh()
    env = smdp.env
    truck = env.trucks[0]
    truck.assigned_target = truck.home_depot  # sent home by the fallback, load untouched
    assert truck.load >= truck.capacity
    env._reload_truck(truck, {"reloads": []}, truck.home_depot)
    assert truck.assigned_target is None, "full-load depot arrival must end the assignment"


def test_hybrid_greedy_no_double_assignment_and_early_termination():
    """End-to-end regression for Probe A: with the fixes, a greedy episode assigns each request to
    exactly one truck, strands no truck on a zero-demand target, and terminates well before the
    horizon (previously every episode ran to max_ticks with a zombie truck orbiting)."""
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=_cfg())
    policy = hybrid_greedy_policy(smdp)
    event = smdp.reset_decision_env()
    env = smdp.env
    assignments: list = []
    steps = 0
    while not event.done and steps < 30000:
        steps += 1
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            pre = {tid: t.assigned_target for tid, t in env.trucks.items()}
            actions = policy(event)
            for tid, node in actions.items():
                if pre.get(tid) is None and node in set(env.assignment_demand):
                    assignments.append((tid, node))
            event, _ = smdp.step_protagonist(actions)
        elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
            event, _ = smdp.step_antagonist(None)
        else:
            event = smdp.advance_until_decision()

    assigned_nodes = [n for _, n in assignments]
    assert len(assigned_nodes) == len(set(assigned_nodes)), f"double assignment: {assignments}"
    for truck in env.trucks.values():
        tgt = truck.assigned_target
        assert tgt is None or tgt == truck.home_depot or env.graph.nodes[tgt]["demand"] > 0, (
            f"truck {truck.truck_id} stranded on served node {tgt}")
    assert env.is_done() and env.time < smdp.config.max_ticks, (
        f"episode must terminate early once served (ended at tick {env.time})")
    delivered = sum(t.delivered_total for t in env.trucks.values())
    assert delivered == len(env.assignment_demand)


def test_hybrid_observation_and_goal_features():
    smdp = _fresh()
    env = smdp.env
    target = "46"
    env.trucks[0].assigned_target = target
    obs = env.observe()
    # static hybrid env ships the queue/ETA/goal blocks (expose_queue_features=True)
    assert "truck_etas" in obs and "goal_dists" in obs
    assert obs["trucks"][0]["assigned_target"] == target
    assert 0 in obs["goal_dists"] and obs["goal_dists"][0][target] == 0.0

    node_ids = sorted(obs["nodes"].keys())
    data = featurize_state(obs, active_truck_id=0)
    assert data.x.shape[1] == NODE_FEATURE_DIM == 14
    tcol = data.x[:, 11]
    assert float(tcol.sum()) == 1.0 and float(tcol[node_ids.index(target)]) == 1.0, (
        "column 11 must mark exactly the active truck's assigned target")
    gcol = data.x[:, 12]
    assert float(gcol[node_ids.index(target)]) == 0.0  # zero distance at the goal itself
    assert float(gcol.max()) > 0.0                     # positive distances elsewhere
    assert float(gcol.max()) <= 10.0 + 1e-6            # clamped

    # the ANTAGONIST view (no active truck) now sees commitments: col 7 marks the target
    data_a = featurize_state(obs, active_truck_id=None)
    assert float(data_a.x[node_ids.index(target), 7]) == 1.0, (
        "antagonist featurization must see truck commitments")
    assert float(data_a.x[:, 11].abs().max()) == 0.0  # but has no active-truck goal columns


def test_narrow_checkpoint_slicing_compat():
    """A pre-bump (11-node / 2-edge dim) agent must consume current 13/4-wide features: sliced
    inputs reproduce the old featurization exactly (new columns are appended last)."""
    from src.agents.sac import _clip_ea, infer_edge_in_dim

    x = torch.randn(7, 13)
    assert torch.equal(_clip_x(x, 11), x[:, :11])
    assert _clip_x(x, 13) is x  # no-op at matching width
    ea = torch.randn(9, 4)
    assert torch.equal(_clip_ea(ea, 2), ea[:, :2])
    assert _clip_ea(ea, 4) is ea

    agent = ProtagonistSAC(node_in_dim=11, edge_in_dim=2, hidden_dim=16, num_layers=2, heads=2, device="cpu")
    assert infer_node_in_dim(agent.actor.state_dict()) == 11
    assert infer_edge_in_dim(agent.actor.state_dict()) == 2
    smdp = _fresh()
    event_obs = smdp.env.observe()
    event_obs["active_truck"] = 0
    mask = smdp.protagonist_action_mask()
    action = agent.select_action(event_obs, {0: mask.get(0, [])})
    assert 0 in action and action[0] in mask.get(0, [])


def test_edge_occupancy_features_encode_motion():
    """gen04 antagonist-observability fix: a truck traversing an edge appears on that DIRECTED
    edge's feature row (count + progress fraction); the reverse direction stays zero."""
    from src.agents.networks import EDGE_FEATURE_DIM

    smdp = _fresh()
    env = smdp.env
    truck = env.trucks[0]
    start = truck.current_node
    neighbor = sorted(env.graph.neighbors(start), key=repr)[0]
    env.dispatch_truck_edge(0, neighbor)   # truck now mid-edge (start -> neighbor)
    env.step()                             # advance one tick so edge_progress > 0

    obs = env.observe()
    data = featurize_state(obs, active_truck_id=None)  # the ANTAGONIST's view
    assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM == 4

    node_ids = sorted(obs["nodes"].keys())
    idx = {n: i for i, n in enumerate(node_ids)}
    ei = data.edge_index
    fwd = rev = None
    for k in range(ei.shape[1]):
        pair = (int(ei[0, k]), int(ei[1, k]))
        if pair == (idx[start], idx[neighbor]):
            fwd = data.edge_attr[k]
        elif pair == (idx[neighbor], idx[start]):
            rev = data.edge_attr[k]
    assert fwd is not None and rev is not None
    if env.trucks[0].edge is not None:  # still mid-edge (edge could be short)
        assert float(fwd[2]) == 1.0, "occupancy count must mark the travel direction"
        assert 0.0 < float(fwd[3]) <= 1.0, "progress fraction must be populated"
        assert float(rev[2]) == 0.0 and float(rev[3]) == 0.0, "reverse direction stays empty"
    # all trucks idle -> all occupancy columns zero
    smdp2 = _fresh()
    d2 = featurize_state(smdp2.env.observe(), active_truck_id=None)
    assert float(d2.edge_attr[:, 2].abs().max()) == 0.0
    assert float(d2.edge_attr[:, 3].abs().max()) == 0.0
