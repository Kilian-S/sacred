"""Step-1 tests for dynamic (Poisson) demand arrivals — the Stage-1.5 env mechanic.

Covers: seeded reproducibility, arrival-rate calibration, reset hygiene, dynamic termination
(only at max_time, never on a transient empty queue), and the latency-telescoping invariant
that the potential-based reward relies on.
"""

from __future__ import annotations

import numpy as np

from src.envs.assignment_factory import make_dynamic_assign_env, poisson_arrival_fn


def test_schedule_reproducible_under_seed():
    a = make_dynamic_assign_env(max_time=400, arrival_rate=0.05)
    b = make_dynamic_assign_env(max_time=400, arrival_rate=0.05)
    a.reset(demand_seed=7)
    b.reset(demand_seed=7)
    assert a._arrival_schedule == b._arrival_schedule
    assert len(a._arrival_schedule) > 0
    # A different seed gives a different schedule (with overwhelming probability).
    c = make_dynamic_assign_env(max_time=400, arrival_rate=0.05)
    c.reset(demand_seed=8)
    assert c._arrival_schedule != a._arrival_schedule


def test_arrival_count_matches_rate():
    rate, horizon = 0.05, 1000
    env = make_dynamic_assign_env(max_time=horizon, arrival_rate=rate)
    counts = []
    for seed in range(40):
        env.reset(demand_seed=seed)
        counts.append(len(env._arrival_schedule))
    # E[count] = rate * horizon = 50; the sample mean should land near it.
    assert abs(np.mean(counts) - rate * horizon) < 6.0


def test_arrivals_land_in_hotspots_only():
    env = make_dynamic_assign_env(max_time=600, arrival_rate=0.1)
    env.reset(demand_seed=1)
    hotspots = set(env.dynamic_hotspots)
    assert all(node in hotspots for _tick, node, _size in env._arrival_schedule)
    assert all(1 <= tick <= 600 for tick, _node, _size in env._arrival_schedule)


def test_reset_clears_dynamic_state():
    env = make_dynamic_assign_env(max_time=300, arrival_rate=0.1)
    env.reset(demand_seed=2)
    for _ in range(150):
        env.step()  # trucks idle at depot; arrivals accumulate undelivered
    assert env.remaining_demand > 0
    env.reset(demand_seed=2)
    assert env.remaining_demand == 0.0
    assert env._arrival_index == 0
    assert env._pending_arrivals == {}
    assert env._delivered_latencies == []


def test_dynamic_terminates_only_at_max_time():
    env = make_dynamic_assign_env(max_time=200, arrival_rate=0.05)
    env.reset(demand_seed=4)
    # At t=0 the queue is empty (no demand yet) but the episode must NOT be done — under the
    # static rule (remaining==0 and trucks home) it would terminate immediately.
    assert env.remaining_demand == 0.0
    assert not env.is_done()
    done_tick = None
    for _ in range(200):
        result = env.step()
        if result.done:
            done_tick = env.time
            break
    assert done_tick == 200


def test_latency_telescoping_invariant():
    """With no deliveries, Σ_t remaining_demand == Σ_requests (T − arrival_tick + 1).

    This is exactly the telescoping the potential-based latency reward depends on: a unit
    outstanding from its arrival tick to the horizon contributes its full wait. Validates
    injection timing and outstanding-count accounting end to end.
    """
    horizon = 250
    env = make_dynamic_assign_env(max_time=horizon, arrival_rate=0.1)
    env.reset(demand_seed=11)
    schedule = list(env._arrival_schedule)
    running = 0.0
    for _ in range(horizon):
        env.step()  # never dispatch -> nothing delivered
        running += env.remaining_demand
    expected = sum(size * (horizon - tick + 1) for tick, _node, size in schedule)
    assert running == expected
    # Nothing delivered -> no recorded latencies, and everything is still outstanding.
    assert env._delivered_latencies == []
    assert env.remaining_demand == sum(size for _t, _n, size in schedule)


def test_delivered_latency_recorded():
    """Drive one truck to a known arrival and check delivered latency = delivery − arrival."""
    # Single depot + single hotspot so the route is deterministic; small graph slice via the
    # real factory but forcing one hotspot keeps the assertion simple.
    env = make_dynamic_assign_env(
        max_time=400, arrival_rate=0.05, depots=("110",), hotspot_nodes=("237",)
    )
    env.reset(demand_seed=5)
    assert env._arrival_schedule, "expected at least one arrival"
    first_tick, node, _size = env._arrival_schedule[0]
    # Step until the request has arrived, then dispatch the truck to serve it.
    while env.time < first_tick:
        env.step()
    assert env.remaining_demand >= 1.0
    env.step(dispatch_actions={0: node})  # route via Dijkstra to the request
    # Run until the delivery is recorded.
    for _ in range(400):
        if env._delivered_latencies:
            break
        env.step()
    assert env._delivered_latencies, "truck never delivered the request"
    assert env._delivered_latencies[0] >= 0


def test_featurize_dynamic_columns_populated():
    """The 2 new node-feature columns (age, ETA) are wired through to the GNN input."""
    from src.agents.networks import featurize_state, NODE_FEATURE_DIM

    env = make_dynamic_assign_env(max_time=400, arrival_rate=0.1)
    env.reset(demand_seed=5)
    while env.remaining_demand < 1:  # wait for the first arrival
        env.step()
    for _ in range(15):             # let it age (trucks sit idle at their depots)
        env.step()
    obs = env.observe()
    assert "node_waits" in obs and "truck_etas" in obs
    assert any(w > 0 for w in obs["node_waits"].values())

    data = featurize_state(obs, active_truck_id=0)
    assert data.x.shape[1] == NODE_FEATURE_DIM == 14
    assert float(data.x[:, 9].max()) > 0.0   # some node has a positive wait (age)
    assert float(data.x[:, 10].max()) > 0.0  # active truck has positive ETAs to demand nodes


def test_featurize_static_has_zero_dynamic_columns():
    """Static non-hybrid problems omit the dynamic keys → the queue/goal columns are exactly zero."""
    from src.agents.networks import featurize_state
    from src.envs.assignment_factory import make_assignment_env

    env = make_assignment_env()
    obs = env.observe()
    assert "node_waits" not in obs and "truck_etas" not in obs
    data = featurize_state(obs, active_truck_id=0)
    assert data.x.shape[1] == 14
    assert float(data.x[:, 9].abs().max()) == 0.0
    assert float(data.x[:, 10].abs().max()) == 0.0
    assert float(data.x[:, 11].abs().max()) == 0.0
    assert float(data.x[:, 12].abs().max()) == 0.0


def _dyn_cfg(max_ticks=300):
    from src.env.smdp_wrapper import SMDPConfig
    return SMDPConfig(
        max_ticks=max_ticks, reward_mode="latency", routing_mode="destination",
        antagonist_interval=20, congestion_duration=30, congestion_budget=200.0,
        congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(0.25, 0.5, 0.75, 1.0))


def test_run_episode_dynamic_metrics():
    """run_episode counts arrivals correctly for dynamic demand (the initial graph is empty)."""
    from src.env.smdp_wrapper import SMDPDecisionWrapper
    from src.baselines.greedy_dispatch import greedy_insertion_policy, no_antagonist_policy, run_episode

    cfg = _dyn_cfg(300)
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_dynamic_assign_env(max_time=300, arrival_rate=0.1, demand_seed=0), config=cfg)
    m = run_episode(smdp, greedy_insertion_policy(smdp), no_antagonist_policy)
    assert m["total_wait"] >= 0
    assert m["num_requests"] == m["delivered"] + int(round(smdp.env.remaining_demand))
    assert m["num_requests"] > 0
    if m["delivered"] > 0:
        assert m["mean_delivered_latency"] >= 0


def test_eval_dynamic_cells_deterministic_and_structured():
    """The multi-seed fixed-adversary eval is reproducible and gap = learned − greedy."""
    import torch
    from scripts.evaluate_dynamic_assign import eval_dynamic_cells, _new_protag, _new_antag, make_env_for_seed_fn

    cfg = _dyn_cfg(150)
    torch.manual_seed(0)
    protag, antag = _new_protag(cfg), _new_antag(cfg)
    mk = make_env_for_seed_fn(arrival_rate=0.1)
    r1 = eval_dynamic_cells(protag, antag, mk, cfg, seeds=(0, 1))
    r2 = eval_dynamic_cells(protag, antag, mk, cfg, seeds=(0, 1))
    assert r1 == r2  # fixed seeds + deterministic (eval-mode) policies -> identical
    for key in ["greedy_atk", "learned_atk", "gap_atk", "gap_noatk"]:
        assert f"{key}_mean" in r1 and f"{key}_std" in r1
        assert r1[f"{key}_std"] >= 0.0
    assert abs((r1["learned_atk_mean"] - r1["greedy_atk_mean"]) - r1["gap_atk_mean"]) < 1e-6


def test_urgency_dispatch_no_double_assignment_and_delivers():
    """The urgency dispatcher runs an episode, delivers, and never assigns two trucks the same
    demand node in a single decision (sequential claiming)."""
    from src.env.smdp_wrapper import SMDPDecisionWrapper, DecisionType
    from src.baselines.greedy_dispatch import urgency_dispatch_policy, no_antagonist_policy, run_episode

    cfg = _dyn_cfg(300)
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_dynamic_assign_env(max_time=300, arrival_rate=0.12, demand_seed=1), config=cfg)
    pol = urgency_dispatch_policy(smdp)
    event = smdp.reset_decision_env()
    saw_multi = False
    while not event.done:
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            actions = pol(event)
            demand_targets = [n for tid, n in actions.items()
                              if smdp.env.graph.nodes[n]["demand"] > 0.0]
            if len(demand_targets) >= 2:
                saw_multi = True
                assert len(demand_targets) == len(set(demand_targets))  # no two trucks -> same request
            event, _ = smdp.step_protagonist(actions)
        elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
            event, _ = smdp.step_antagonist(None)
        else:
            event = smdp.advance_until_decision()
    assert len(smdp.env._delivered_latencies) > 0  # actually delivered something
    # And it composes through run_episode with a finite latency.
    s2 = SMDPDecisionWrapper(
        env_factory=lambda: make_dynamic_assign_env(max_time=300, arrival_rate=0.12, demand_seed=1), config=cfg)
    assert run_episode(s2, urgency_dispatch_policy(s2), no_antagonist_policy)["total_wait"] >= 0


def test_antag_per_event_cap_limits_congestions():
    """max_antag_actions_per_event=1 -> the adversary congests at most one edge per decision event
    (the sequential epoch stops after one sub-action)."""
    from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper, DecisionType

    cfg = SMDPConfig(
        max_ticks=300, reward_mode="latency", routing_mode="destination",
        antagonist_interval=25, congestion_duration=120, congestion_budget=4000.0,
        congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(1.0,),
        max_antag_actions_per_event=1)
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_dynamic_assign_env(max_time=300, arrival_rate=0.1, demand_seed=2), config=cfg)

    # Always-attack adversary: congest the first allowed edge at full level.
    def attacker(event):
        lbe = event.antagonist_action_mask.get("levels_by_edge", {})
        if not lbe:
            return None
        e = sorted(lbe, key=repr)[0]
        return (e, max(lbe[e]))

    event = smdp.reset_decision_env()
    max_per_event = 0
    while not event.done:
        if event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            before = dict(smdp.active_congestion)
            event, _ = smdp.step_antagonist(attacker(event))
            # count newly-congested edges attributable to this single decision event
            added = len(set(smdp.active_congestion) - set(before))
            max_per_event = max(max_per_event, added)
        elif event.decision_type == DecisionType.PROTAGONIST_DECISION:
            event, _ = smdp.step_protagonist({})
        else:
            event = smdp.advance_until_decision()
    assert max_per_event <= 1, f"cap=1 but saw {max_per_event} congestions in one event"


def test_full_block_antagonist_attacks_and_updates():
    """Regression for two latent hardcoded-level bugs exposed by congestion_levels=(1.0,):
    (a) select_action must return level 1.0 (in the mask), not the hardcoded 0.25 -> else every
        attack is silently rejected and budget stays 0;
    (b) the antagonist update must map the level value back to the right index -> else an IndexError
        crashes the antagonist phase.
    """
    from src.env.smdp_wrapper import SMDPDecisionWrapper, DecisionType
    from scripts.evaluate_dynamic_assign import dynassign_config, _new_antag

    cfg = dynassign_config()
    assert cfg.congestion_levels == (1.0,)
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_dynamic_assign_env(arrival_rate=0.08, demand_seed=0, max_time=300), config=cfg)
    antag = _new_antag(cfg)

    ev = smdp.reset_decision_env()
    chosen_levels = []
    while not ev.done:
        if ev.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            mask = ev.antagonist_action_mask
            rem = smdp.budget.remaining
            a = antag.select_action(ev.observation, mask, rem, deterministic=False)
            if a is not None:
                chosen_levels.append(a[1])
            ev, tr = smdp.step_antagonist(a)
            # Minimal trainer-style enrichment so the update can parse the transition (guards (b)).
            st = dict(tr.state)
            st["allowed_destinations"] = {"antagonist": {
                "allowed_edges": list(mask.get("levels_by_edge", {}).keys()),
                "original_edges": list(tr.state["edges"].keys())}}
            tr.state = st
            ns = dict(tr.next_state)
            ns["allowed_destinations"] = {"antagonist": {
                "allowed_edges": list(ev.antagonist_action_mask.get("levels_by_edge", {}).keys()),
                "original_edges": list(tr.next_state["edges"].keys())}}
            tr.next_state = ns
            tr.info["antagonist_budget_remaining"] = rem
            tr.info["next_antagonist_budget_remaining"] = smdp.budget.remaining
            antag.replay_buffer.push(tr)
        elif ev.decision_type == DecisionType.PROTAGONIST_DECISION:
            ev, _ = smdp.step_protagonist({})
        else:
            ev = smdp.advance_until_decision()

    assert chosen_levels, "antagonist never chose to attack"
    assert all(lvl == 1.0 for lvl in chosen_levels), f"expected only level 1.0, got {set(chosen_levels)}"
    assert smdp.budget.used > 0, "attacks were rejected -> budget unspent (the select_action bug)"
    antag.update(batch_size=8)  # must not IndexError (the update level-index bug)
