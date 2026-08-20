#!/usr/bin/env python3
"""Generate expert replay-buffer demonstrations for the assignment probe: greedy-insertion
episodes with the antagonist off, collected through the shared transition builder so the
transitions are byte-identical to live training transitions.

    PYTHONPATH=. python scripts/generate_erb_assign.py --episodes 100 --out data/erb_assign.pt
"""

from __future__ import annotations

import argparse

import torch

from src.env.smdp_wrapper import DecisionType, SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_assignment_env
from src.agents.transition_builder import collect_protagonist_transitions
from src.baselines.greedy_dispatch import _congestion_aware_distance, _id_key


def assignment_config() -> SMDPConfig:
    """SMDP configuration for the assignment probe."""
    return SMDPConfig(
        max_ticks=800, antagonist_interval=20, congestion_duration=30,
        congestion_budget=400.0, congestion_cooldown=0, congestion_cost=0.1,
        reward_mode="latency", routing_mode="destination", congestion_levels=(0.25, 0.5, 0.75, 1.0),
    )


def greedy_choose_fn(smdp: SMDPDecisionWrapper):
    """Per-truck greedy-insertion choice: nearest unserved request by congestion-aware distance,
    else the depot. Takes the per-truck form the transition builder expects."""
    def choose(projected_obs, truck_mask, truck_id):
        env = smdp.env
        dests = truck_mask.get(truck_id, [])
        if not dests:
            return {}
        requests = [d for d in dests if env.graph.nodes[d]["demand"] > 0.0]
        source = env.trucks[truck_id].current_node
        if requests and source is not None:
            best = min(requests, key=lambda d: (_congestion_aware_distance(env, source, d), _id_key(d)))
            return {truck_id: best}
        return {truck_id: dests[0]}  # depot: reload / return
    return choose


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate greedy no-attack ERB demos for the assignment probe.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--out", type=str, default="data/erb_assign.pt")
    args = parser.parse_args()

    cfg = assignment_config()
    smdp = SMDPDecisionWrapper(env_factory=lambda: make_assignment_env(), config=cfg)
    choose = greedy_choose_fn(smdp)

    all_transitions: list = []
    for ep in range(args.episodes):
        event = smdp.reset_decision_env()
        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                event, transitions = collect_protagonist_transitions(smdp, event, choose)
                all_transitions.extend(transitions)
            elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                event, _ = smdp.step_antagonist(None)  # no-attack: antagonist does nothing
            else:
                event = smdp.advance_until_decision()
        if (ep + 1) % 20 == 0:
            print(f"  {ep+1}/{args.episodes} episodes, {len(all_transitions)} transitions")

    torch.save(all_transitions, args.out)
    print(f"\nSaved {len(all_transitions)} greedy no-attack protagonist transitions to {args.out}")
    assert all(t.agent == "protagonist" for t in all_transitions)
    t0 = all_transitions[0]
    print(f"sample transition: active_truck={t0.state.get('active_truck')} "
          f"action={t0.action} reward={t0.reward:.1f} "
          f"mask_size={len(t0.action_mask['protagonist'].get(t0.state.get('active_truck'), []))}")


if __name__ == "__main__":
    main()
