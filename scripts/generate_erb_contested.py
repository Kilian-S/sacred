#!/usr/bin/env python3
"""Generate ERB demonstrations for the contested arena (B7, Obj-3 bootstrapping ablation).

Runs greedy-insertion episodes on the contested env WITH THE ANTAGONIST OFF (no-attack demos),
collecting protagonist transitions via the shared transition builder so they are byte-identical
to live training transitions. Seeds the protagonist's replay buffer with competent clean-road
dispatch, giving the B3 curriculum a competent starting policy (the objective's "accelerate
convergence" claim). Each episode draws a fresh Poisson demand stream (reproducible via
--demand-seed) so the demos span demand realisations.

    PYTHONPATH=. python scripts/generate_erb_contested.py --episodes 100 --out data/erb_contested.pt

Load into training with: --erb-path data/erb_contested.pt (existing flag).
"""

from __future__ import annotations

import argparse
import itertools

import torch

from src.env.smdp_wrapper import DecisionType, SMDPDecisionWrapper
from src.envs.contested import contested_config, make_contested_env
# Reuse the assignment ERB's per-truck greedy choice (generic greedy insertion, mask-driven).
from scripts.generate_erb_assign import greedy_choose_fn


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate greedy no-attack ERB demos for the contested arena.")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--arrival-rate", type=float, default=0.06)
    parser.add_argument("--demand-seed", type=int, default=777, help="base seed; advanced per episode")
    parser.add_argument("--out", type=str, default="data/erb_contested.pt")
    args = parser.parse_args()

    cfg = contested_config()
    counter = itertools.count(args.demand_seed)
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_contested_env(arrival_rate=args.arrival_rate, demand_seed=next(counter)),
        config=cfg)
    choose = greedy_choose_fn(smdp)

    all_transitions: list = []
    for ep in range(args.episodes):
        event = smdp.reset_decision_env()
        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                event, transitions = collect(smdp, event, choose)
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


def collect(smdp, event, choose):
    from src.agents.transition_builder import collect_protagonist_transitions
    return collect_protagonist_transitions(smdp, event, choose)


if __name__ == "__main__":
    main()
