#!/usr/bin/env python3
"""Exploitability BR gate: trains a best-response attacker against a frozen greedy protagonist,
producing an antagonist checkpoint that evaluate_portfolio can compare against random and the
reactive scripted attacker. Arena: `contested` (dynassign dynamics + route reach, twin reward) or
`hybrid` (static, no twin reward).

    PYTHONPATH=. python scripts/br_gate.py --arena contested --reward-baseline twin \
        --antag-target-entropy 0.5 --gamma 0.997 --episodes 300 --tag br_vs_greedy_fixed
    PYTHONPATH=. python scripts/evaluate_portfolio.py --problem contested \
        --br gate=models/runs/br_gate/br_vs_greedy_fixed_seed0/antagonist/actor.pt \
        --attackers none,random,pathrand,targeted,br_gate --instances 24 --seed-base 20000019
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools

from src.agents.sac import AntagonistSAC, ProtagonistSAC
from src.agents.sacred_atla import ATLACoevolutionTrainer
from src.env.smdp_wrapper import SMDPDecisionWrapper


def main() -> None:
    p = argparse.ArgumentParser(description="Train a best-response attacker vs frozen greedy (BR gate).")
    p.add_argument("--arena", choices=["contested", "hybrid"], default="contested")
    p.add_argument("--reward-baseline", choices=["none", "twin"], default="twin")
    p.add_argument("--antag-target-entropy", type=float, default=0.5,
                   help="absolute antagonist entropy target (entropy-pinning fix); None-like high default is 0.5*ln(N)")
    p.add_argument("--gamma", type=float, default=0.997)
    p.add_argument("--arrival-rate", type=float, default=0.06, help="contested only")
    p.add_argument("--episodes", type=int, default=300)
    p.add_argument("--switch-every", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--group", type=str, default="br_gate")
    p.add_argument("--tag", type=str, default="br_vs_greedy_fixed")
    args = p.parse_args()

    import random
    import numpy as np
    import torch
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    baseline_provider = None
    if args.arena == "contested":
        from src.envs.contested import contested_config, make_contested_env, make_greedy_twin_baseline_provider
        from scripts.generate_erb_assign import greedy_choose_fn
        config = contested_config()
        if args.reward_baseline == "twin":
            config = dataclasses.replace(config, reward_baseline="twin")
        counter = itertools.count(args.seed * 100003)
        env_factory = lambda: make_contested_env(arrival_rate=args.arrival_rate, demand_seed=next(counter))
        if args.reward_baseline == "twin":
            baseline_provider = make_greedy_twin_baseline_provider(config, arrival_rate=args.arrival_rate)
        chooser_of = greedy_choose_fn  # destination-mode per-truck greedy
    else:
        from scripts.evaluate_hybrid import hybrid_config
        from src.envs.assignment_factory import make_hybrid_assign_env
        from src.baselines.greedy_dispatch import hybrid_greedy_chooser
        if args.reward_baseline == "twin":
            raise SystemExit("twin reward is not wired for the hybrid arena; use --arena contested.")
        config = hybrid_config()
        env_factory = make_hybrid_assign_env
        chooser_of = hybrid_greedy_chooser

    smdp = SMDPDecisionWrapper(env_factory=env_factory, config=config, baseline_provider=baseline_provider)

    # Dummy protagonist net (required by the constructor, never queried: the greedy chooser overrides).
    protag = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4, device="cpu")
    antag = AntagonistSAC(
        node_in_dim=13, edge_in_dim=4, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(config.congestion_levels),
        level_costs=[lvl * config.congestion_duration for lvl in config.congestion_levels],
        congestion_levels=config.congestion_levels,
        gamma=args.gamma, target_entropy=args.antag_target_entropy, device="cpu")

    print(f"BR gate (corrected): arena={args.arena} reward_baseline={args.reward_baseline} "
          f"antag_target_entropy={args.antag_target_entropy} gamma={args.gamma}; "
          f"training antagonist vs frozen GREEDY for {args.episodes} ep.")
    trainer = ATLACoevolutionTrainer(
        smdp=smdp, protag_agent=protag, antag_agent=antag,
        switch_every_episodes=args.switch_every, batch_size=args.batch_size,
        run_name=f"{args.group}/{args.tag}_seed{args.seed}",
        mode="antagonist_only",
        frozen_protagonist_chooser=chooser_of(smdp),
    )
    trainer.run_training(total_episodes=args.episodes)


if __name__ == "__main__":
    main()
