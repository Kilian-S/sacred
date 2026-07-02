"""Non-learning baselines and an episode-eval harness for the Stage-0 rung.

The greedy dispatcher is the reference the learned protagonist must beat: at every
decision it sends each idle truck to the *nearest* allowed destination along the
**congestion-aware** shortest path (``effective_weight``, which already encodes the
antagonist's congestion). For the capacity-1 Stage-0 shuttle this is greedy
nearest-request scheduling — the natural, strong, non-adaptive heuristic.

``run_episode`` drives the SMDP wrapper with arbitrary protagonist/antagonist
*policies* (callables of the current ``DecisionEvent``), so the same harness scores
the greedy baseline and a trained SAC agent under the same antagonist. It mirrors the
decision-branch structure of :class:`ATLACoevolutionTrainer` exactly.
"""

from __future__ import annotations

from typing import Any, Callable

import networkx as nx

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import DecisionEvent, DecisionType, SMDPDecisionWrapper

# A policy maps the current decision event to an action (or no-op).
ProtagPolicy = Callable[[DecisionEvent], "dict[int, Any]"]
AntagPolicy = Callable[[DecisionEvent], "tuple | None"]


def _id_key(node_id: Any) -> tuple:
    s = str(node_id)
    return (0, int(s)) if s.isdigit() else (1, s)


def _congestion_aware_distance(env: GraphEnv, source: Any, dest: Any) -> float:
    """Shortest-path length from ``source`` to ``dest`` under current congestion."""
    try:
        return nx.dijkstra_path_length(env.graph, source, dest, weight="effective_weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return float("inf")


def greedy_protagonist_action(env: GraphEnv, action_mask: "dict[int, list[Any]]") -> "dict[int, Any]":
    """Send each idle truck to its nearest allowed destination (congestion-aware)."""
    actions: dict[int, Any] = {}
    for truck_id, destinations in action_mask.items():
        if not destinations:
            continue
        source = env.trucks[truck_id].current_node
        if source is None:
            continue
        best = min(
            destinations,
            key=lambda d: (_congestion_aware_distance(env, source, d), _id_key(d)),
        )
        actions[truck_id] = best
    return actions


def greedy_protagonist_policy(smdp: SMDPDecisionWrapper) -> ProtagPolicy:
    """A protagonist policy callable that greedily dispatches to the nearest request."""
    return lambda event: greedy_protagonist_action(smdp.env, event.protagonist_action_mask)


def _greedy_goal(env: GraphEnv, truck) -> Any:
    """The node a reactive dispatcher heads for: nearest outstanding customer if the truck
    has load, else the home depot to reload."""
    if truck.load > 0:
        customers = [
            n for n, d in env.graph.nodes(data=True)
            if not d.get("has_depot", False) and d.get("demand", 0.0) > 0.0
        ]
        if customers:
            return min(
                customers,
                key=lambda c: (_congestion_aware_distance(env, truck.current_node, c), _id_key(c)),
            )
    return truck.home_depot


def greedy_next_hop_action(env: GraphEnv, action_mask: "dict[int, list[Any]]") -> "dict[int, Any]":
    """Reactive next-hop greedy: step onto the first hop of the congestion-aware shortest
    path to the current goal. This reacts to congestion *now* but cannot anticipate the
    antagonist — the headroom a learned policy can exploit."""
    actions: dict[int, Any] = {}
    for truck_id, neighbors in action_mask.items():
        if not neighbors:
            continue
        truck = env.trucks[truck_id]
        source = truck.current_node
        goal = _greedy_goal(env, truck)
        nxt = None
        if goal is not None and goal != source:
            try:
                path = nx.dijkstra_path(env.graph, source, goal, weight="effective_weight")
                if len(path) > 1:
                    nxt = path[1]
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                nxt = None
        actions[truck_id] = nxt if nxt in neighbors else neighbors[0]
    return actions


def greedy_next_hop_policy(smdp: SMDPDecisionWrapper) -> ProtagPolicy:
    """A next-hop protagonist policy callable (reactive congestion-aware shortest path)."""
    return lambda event: greedy_next_hop_action(smdp.env, event.protagonist_action_mask)


def greedy_insertion_policy(smdp: SMDPDecisionWrapper) -> ProtagPolicy:
    """Multi-truck greedy-insertion baseline (destination mode): each free truck takes the
    nearest (congestion-aware) unserved request, with *sequential claiming* so two trucks
    deciding in the same event never grab the same request. Depot (reload/return) only if no
    request is available. This is the reactive classical dispatcher the learned policy must beat.
    """
    def policy(event: DecisionEvent) -> "dict[int, Any]":
        env = smdp.env
        actions: dict[int, Any] = {}
        claimed: set = set()
        for truck_id in sorted(event.protagonist_action_mask):
            dests = [d for d in event.protagonist_action_mask[truck_id] if d not in claimed]
            if not dests:
                continue
            requests = [d for d in dests if env.graph.nodes[d]["demand"] > 0.0]
            source = env.trucks[truck_id].current_node
            if requests and source is not None:
                best = min(requests, key=lambda d: (_congestion_aware_distance(env, source, d), _id_key(d)))
                actions[truck_id] = best
                claimed.add(best)
            else:
                actions[truck_id] = dests[0]  # depot: reload / return
        return actions

    return policy


def urgency_dispatch_policy(smdp: SMDPDecisionWrapper) -> ProtagPolicy:
    """Oldest-first (most-urgent) dispatcher: each free truck takes the longest-waiting unserved
    request, tie-broken by congestion-aware distance, with sequential claiming. Unlike greedy-
    insertion (nearest), this *uses request age* — the signal greedy is blind to — so it tests
    whether prioritising the backlog beats myopic-nearest under load. Reactive (no anticipation),
    so it is a heuristic floor on "smart", not a ceiling. (Dynamic-demand only; reads the env's
    per-node pending-arrival ages.)"""
    def policy(event: DecisionEvent) -> "dict[int, Any]":
        env = smdp.env
        pending = getattr(env, "_pending_arrivals", {})
        ages = {node: env.time - dq[0] for node, dq in pending.items() if dq}
        actions: dict[int, Any] = {}
        claimed: set = set()
        for truck_id in sorted(event.protagonist_action_mask):
            dests = [d for d in event.protagonist_action_mask[truck_id] if d not in claimed]
            if not dests:
                continue
            requests = [d for d in dests if env.graph.nodes[d]["demand"] > 0.0]
            source = env.trucks[truck_id].current_node
            if requests and source is not None:
                # max age first; tie-break nearest (congestion-aware), then id for determinism.
                best = min(
                    requests,
                    key=lambda d: (-ages.get(d, 0.0), _congestion_aware_distance(env, source, d), _id_key(d)),
                )
                actions[truck_id] = best
                claimed.add(best)
            else:
                actions[truck_id] = dests[0]  # depot: reload / return
        return actions
    return policy


def hybrid_greedy_policy(smdp: SMDPDecisionWrapper) -> ProtagPolicy:
    """Hybrid baseline (`routing_mode="hybrid"`): the strong REACTIVE dispatcher the learned policy
    must beat. **Assignment** — each unassigned truck takes the nearest (congestion-aware) unclaimed
    request, with sequential claiming. **Routing** — each assigned truck steps to the forward
    next-hop on the congestion-aware shortest path to its target. It reroutes around congestion it
    can *see now*, but cannot *anticipate* the adversary — the headroom a learned policy can exploit.
    """
    def policy(event: DecisionEvent) -> "dict[int, Any]":
        env = smdp.env
        actions: dict[int, Any] = {}
        claimed: set = set()
        for truck_id in sorted(event.protagonist_action_mask):
            options = event.protagonist_action_mask[truck_id]
            if not options:
                continue
            truck = env.trucks[truck_id]
            source = truck.current_node
            if source is None:
                continue
            if truck.assigned_target is None:
                # assignment: nearest unclaimed request (candidates here are demand nodes)
                avail = [n for n in options if n not in claimed]
                if not avail:
                    continue
                best = min(avail, key=lambda d: (_congestion_aware_distance(env, source, d), _id_key(d)))
                actions[truck_id] = best
                claimed.add(best)
            else:
                # routing: forward next-hop on the congestion-aware shortest path to the target
                try:
                    path = nx.dijkstra_path(env.graph, source, truck.assigned_target, weight="effective_weight")
                    nxt = path[1] if len(path) > 1 else options[0]
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    nxt = options[0]
                actions[truck_id] = nxt if nxt in options else options[0]
        return actions

    return policy


def no_antagonist_policy(event: DecisionEvent) -> None:
    """Antagonist that never congests (clean, no-attack baseline)."""
    return None


def run_episode(
    smdp: SMDPDecisionWrapper,
    protag_policy: ProtagPolicy,
    antag_policy: AntagPolicy = no_antagonist_policy,
) -> dict[str, Any]:
    """Run one episode driving the wrapper with the given policies; return metrics.

    Returns a dict with:
      * ``total_wait``    -- telescoped delivery latency = sum over ticks of outstanding
                             request count (= ``-protagonist_reward`` in latency mode).
                             Lower is better; includes still-undelivered requests up to
                             the horizon. **This is the headline comparison metric.**
      * ``delivered`` / ``num_requests`` / ``delivery_rate``
      * ``mean_completion_tick`` -- mean delivery tick over *delivered* requests only.
      * ``ticks`` / ``budget_used``
    """
    event = smdp.reset_decision_env()
    # Total demand units to deliver, from the pristine initial graph (robust to any auto-
    # advancement during reset in next-hop mode).
    num_requests = int(round(sum(d.get("demand", 0.0) for _, d in smdp.env._initial_graph.nodes(data=True))))

    ep_protag_reward = 0.0
    delivery_ticks: list[int] = []

    def _scan_deliveries(ev: DecisionEvent) -> None:
        for tick_info in ev.info.get("events", []):
            for delivery in tick_info.get("deliveries", []):
                delivery_ticks.append(tick_info["time"])

    _scan_deliveries(event)

    while not event.done:
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            actions = protag_policy(event)
            next_event, transition = smdp.step_protagonist(actions)
            ep_protag_reward += transition.reward
        elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            action = antag_policy(event)
            next_event, transition = smdp.step_antagonist(action)
            ep_protag_reward += next_event.protagonist_reward
        else:
            next_event = smdp.advance_until_decision()

        _scan_deliveries(next_event)
        event = next_event

    delivered = len(delivery_ticks)
    # Dynamic demand: the initial graph is empty (demand arrives over time), so count the requests
    # that actually entered = delivered + still-queued, and report the mean wait of completed
    # requests from the env's per-request latency log.
    mean_delivered_latency = float("nan")
    if getattr(smdp.env, "_dynamic_demand", False):
        delivered_lat = smdp.env._delivered_latencies
        num_requests = len(delivered_lat) + int(round(smdp.env.remaining_demand))
        if delivered_lat:
            mean_delivered_latency = sum(delivered_lat) / len(delivered_lat)
    return {
        "total_wait": -ep_protag_reward,
        "delivered": delivered,
        "num_requests": num_requests,
        "delivery_rate": delivered / max(1, num_requests),
        "mean_completion_tick": (sum(delivery_ticks) / delivered) if delivered else float("nan"),
        "mean_delivered_latency": mean_delivered_latency,
        "ticks": smdp.env.time,
        "budget_used": smdp.budget.used,
    }
