#!/usr/bin/env python3
"""Exploitability BR gate: train a best-response attacker against a FROZEN GREEDY victim on the
hybrid route-reach arena, then (separately, via evaluate_portfolio) check it beats random + the
reactive scripted attacker.

The pivotal go/no-go for the gen07 exploitability direction (2026-07-06). The greedy probes showed
crude unpredictability vs a REACTIVE attacker is thin/negative; the real question is whether an
ANTICIPATORY best-response attacker can exploit a competent deterministic policy's predictability.
gen05 showed transferred BRs hit greedy for +1667 (> scripted) in this arena but never TRAINED a
BR against greedy; this does. Uses the existing (flat) antagonist head + route-reach mask (gen05
showed the mask does the aiming there, so the factored head is not needed unless this comes in
marginal).

    PYTHONPATH=. python scripts/br_gate.py --episodes 300 --tag br_vs_greedy --group br_gate
    # then eval:
    PYTHONPATH=. python scripts/evaluate_portfolio.py --problem hybrid \
        --br gate=models/runs/br_gate/br_vs_greedy_seed0/antagonist/actor.pt \
        --attackers none,random,targeted,gateway,br_gate --instances 24 --seed-base 20000019
"""

from __future__ import annotations

import argparse

from scripts.evaluate_hybrid import hybrid_config
from src.agents.sac import AntagonistSAC, ProtagonistSAC
from src.agents.sacred_atla import ATLACoevolutionTrainer
from src.baselines.greedy_dispatch import hybrid_greedy_chooser
from src.env.smdp_wrapper import SMDPDecisionWrapper
from src.envs.assignment_factory import make_hybrid_assign_env


def main() -> None:
    p = argparse.ArgumentParser(description="Train a best-response attacker vs frozen greedy (BR gate).")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--switch-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--group", type=str, default="br_gate")
    p.add_argument("--tag", type=str, default="br_vs_greedy")
    args = p.parse_args()

    import random
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    config = hybrid_config()  # route reach, full block, budget 1500 (gen05 arena)
    smdp = SMDPDecisionWrapper(env_factory=make_hybrid_assign_env, config=config)

    # Frozen victim = greedy (deterministic, competent). A dummy protagonist net is required by the
    # trainer constructor but never queried (the chooser overrides it in antagonist_only mode).
    protag = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4, device="cpu")
    antag = AntagonistSAC(
        node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(config.congestion_levels),
        level_costs=[lvl * config.congestion_duration for lvl in config.congestion_levels],
        congestion_levels=config.congestion_levels, device="cpu")

    trainer = ATLACoevolutionTrainer(
        smdp=smdp, protag_agent=protag, antag_agent=antag,
        switch_every_episodes=args.switch_every, batch_size=args.batch_size,
        run_name=f"{args.group}/{args.tag}_seed{args.seed}",
        mode="antagonist_only",
        frozen_protagonist_chooser=hybrid_greedy_chooser(smdp),
    )
    print(f"BR gate: training antagonist vs frozen GREEDY on hybrid route-reach for {args.episodes} ep.")
    trainer.run_training(total_episodes=args.episodes)


if __name__ == "__main__":
    main()
