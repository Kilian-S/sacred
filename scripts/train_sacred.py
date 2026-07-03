#!/usr/bin/env python3
"""Main training script for the SACRED (Soft Actor-Critic Robust Evolutionary Deep RL) framework.

This script executes coevolutionary training between the protagonist fleet dispatcher
and the antagonist congestion creator from scratch.
"""

import argparse
import sys

from src.env.smdp_wrapper import SMDPDecisionWrapper, SMDPConfig
from src.env.toy_graph import make_toy_graph_env
from src.agents.sac import ProtagonistSAC, AntagonistSAC
from src.agents.sacred_atla import ATLACoevolutionTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SACRED agents with ATLA from scratch.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=1,
        help="Total number of episodes to train (default: 1 for testing)",
    )
    parser.add_argument(
        "--switch-every",
        type=int,
        default=5,
        help="Alternate training phases every N episodes (default: 5)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for SAC updates (default: 32)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=64,
        help="Hidden feature size for GATv2 networks (default: 64)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/tb_runs",
        help="Directory to save TensorBoard logs (default: logs/tb_runs)",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default="sacred_atla",
        help="Custom tag for this training run (default: sacred_atla)",
    )
    parser.add_argument(
        "--preseed-buffer",
        type=lambda x: (str(x).lower() == 'true'),
        default=True,
        help="Pre-seed the protagonist's replay buffer using ALNS demonstrations (default: True)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to use for PyTorch GATv2 training: cpu, mps, or cuda (default: cpu)",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to a directory containing protagonist/checkpoint.pt and antagonist/checkpoint.pt to resume from.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=100,
        help="Stage 0 only: run a learned-vs-greedy eval every N episodes (0 disables).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (torch/numpy/random). Set for reproducible, labelled seeded runs in a "
             "generation; None = unseeded (legacy nondeterministic run).",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Experiment generation name. Nests the run under logs/tb_runs/<group>/ and "
             "models/runs/<group>/ so TensorBoard groups it and seeds stay together.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="torch CPU thread cap (default ~4). Lower it (e.g. 3) when running several seeds "
             "in parallel so total threads stay <= 10 cores (see scratch/thread_benchmark.py).",
    )
    parser.add_argument(
        "--erb-path",
        type=str,
        default=None,
        help="Path to a .pt of pre-generated SMDPTransitions to seed the protagonist replay "
             "buffer (e.g. data/erb_assign.pt from generate_erb_assign.py). Overrides the legacy "
             "--preseed-buffer path; demos are already correctly formatted (no compat shim).",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=["osm", "stage0", "assign", "dynassign", "hybrid"],
        default="osm",
        help=(
            "Which problem to train. 'osm' (default) = the static-demand Kaliningrad "
            "baseline. 'stage0' = single-truck next-hop route-choice validation rung. "
            "'assign' = the 3b multi-truck assignment probe (static, RETIRED baseline). "
            "'dynassign' = Stage 1.5 dynamic assignment (Poisson arrivals, 2 trucks, latency). "
            "'hybrid' = Stage 2 hybrid (assignment + next-hop routing, chokepoint geometry, static)."
        ),
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        default=0.06,
        help="dynassign only: Poisson demand arrival rate (requests/tick). Default 0.06 = the "
             "Step-4 gate's rho~1 operating point (delivery ~0.71, residual queue ~14).",
    )
    parser.add_argument(
        "--congestion-budget",
        type=float,
        default=4000.0,
        help="dynassign only: antagonist congestion budget. Default 4000 is non-binding under the "
             "per-event cap (~32 full roadblocks x 120 = ~3840 spent); the cap + congestion_duration "
             "are the real leverage/compute knobs now.",
    )
    parser.add_argument(
        "--vanilla",
        action="store_true",
        help="NON-adversarial control for the robustness comparison: the protagonist trains every "
             "episode and the antagonist is inert (never acts/updates). Identical env, reward, nets "
             "and hyperparameters otherwise.",
    )
    parser.add_argument(
        "--train-antagonist-only",
        action="store_true",
        help="Best-response attacker training: FREEZE the protagonist (loaded from "
             "--protagonist-snapshot, acting stochastically) and train only the antagonist. Used to "
             "build the per-policy worst-case attack for the evaluation portfolio.",
    )
    parser.add_argument(
        "--protagonist-snapshot",
        type=str,
        default=None,
        help="Actor state_dict (.pt) of the frozen protagonist for --train-antagonist-only. The "
             "protagonist nets are sized to the snapshot's trained feature width automatically.",
    )
    args = parser.parse_args()

    if args.vanilla and args.train_antagonist_only:
        sys.exit("--vanilla and --train-antagonist-only are mutually exclusive.")
    if args.train_antagonist_only and not args.protagonist_snapshot:
        sys.exit("--train-antagonist-only requires --protagonist-snapshot.")
    trainer_mode = "vanilla" if args.vanilla else ("antagonist_only" if args.train_antagonist_only else "atla")

    # Reproducibility + CPU thread cap (for parallel seeded runs; see scratch/thread_benchmark.py).
    if args.threads is not None:
        import torch
        torch.set_num_threads(args.threads)
    if args.seed is not None:
        import random
        import numpy as np
        import torch
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        print(f"Seeded run: seed={args.seed}")

    # 1. Initialize the SMDP environment for the chosen problem.
    if args.problem == "stage0":
        print("Initializing SACRED Stage-0 next-hop route-choice validation environment (single truck)...")
        from src.envs.stage0_factory import make_stage0_nexthop_env

        config = SMDPConfig(
            max_ticks=400,
            antagonist_interval=20,  # corridor is short; act often
            congestion_duration=30,
            congestion_budget=300.0,
            congestion_cooldown=0,
            congestion_cost=0.1,
            reward_mode="latency",  # per-tick outstanding-wait; telescopes to total latency
            routing_mode="next_hop",  # policy chooses each edge -> learns to route around the antagonist
            routing_corridor_slack=1.2,  # tightened from 1.5: keeps both routes, prunes the mid-corridor dithering
            congestion_levels=(0.25, 0.5, 0.75, 1.0),
        )
        smdp = SMDPDecisionWrapper(
            env_factory=lambda: make_stage0_nexthop_env(),
            config=config,
        )
        # Stage 0 has no demonstration ERB yet (the dynamic-dispatch ERB is later-stage work).
        if args.preseed_buffer:
            print("Stage 0: forcing --preseed-buffer False (no ERB for the validation rung).")
            args.preseed_buffer = False
        # Next-hop fragments each episode into many short transitions, so the per-transition
        # latency reward is small (~-0.15 at scale 0.01) and was dwarfed by the SAC entropy
        # bonus alpha*H (~0.5) -> the agent optimised entropy, not delivery (Q went positive
        # while delivery collapsed). Scale up ~10x so the task signal dominates the entropy term.
        reward_scale = 0.1
    elif args.problem == "assign":
        print("Initializing SACRED 3b assignment probe (2 trucks/depots, contested demand)...")
        from src.envs.assignment_factory import make_assignment_env

        config = SMDPConfig(
            max_ticks=800,
            antagonist_interval=20,
            congestion_duration=30,
            congestion_budget=400.0,
            congestion_cooldown=0,
            congestion_cost=0.1,
            reward_mode="latency",
            routing_mode="destination",  # assignment only: env auto-routes (exact Dijkstra)
            congestion_levels=(0.25, 0.5, 0.75, 1.0),
        )
        smdp = SMDPDecisionWrapper(
            env_factory=lambda: make_assignment_env(),
            config=config,
        )
        if args.preseed_buffer:
            print("Assign: forcing --preseed-buffer False (no ERB for the probe).")
            args.preseed_buffer = False
        # Same entropy-vs-signal scaling concern as stage0: latency reward must dominate alpha*H.
        reward_scale = 0.1
    elif args.problem == "dynassign":
        print("Initializing SACRED Stage-1.5 dynamic assignment (Poisson arrivals, 2 trucks)...")
        import itertools
        from src.envs.assignment_factory import make_dynamic_assign_env

        config = SMDPConfig(
            max_ticks=800,
            antagonist_interval=25,           # ~32 antagonist decision events / episode
            congestion_duration=120,          # each full roadblock persists ~5 events (sustained)
            congestion_budget=args.congestion_budget,  # default 4000 (non-binding under the cap)
            congestion_cooldown=0,
            congestion_cost=0.1,
            reward_mode="latency",
            routing_mode="destination",       # assignment only: env auto-routes (routing deferred)
            congestion_levels=(1.0,),         # FULL blockage only (simpler adversary + fewer updates)
            max_antag_actions_per_event=1,    # one strategic roadblock per event -> ~32 updates/ep
        )
        # Each episode gets a fresh Poisson demand stream. When --seed is set, derive a distinct
        # per-episode demand seed from a counter so the whole run's stream sequence is reproducible
        # (the env is rebuilt each reset, so a fixed seed would repeat one stream — the counter
        # advances it instead). Unseeded -> OS entropy per episode.
        if args.seed is not None:
            _demand_counter = itertools.count(args.seed * 100003)
            env_factory = lambda: make_dynamic_assign_env(
                arrival_rate=args.arrival_rate, demand_seed=next(_demand_counter))
        else:
            env_factory = lambda: make_dynamic_assign_env(arrival_rate=args.arrival_rate)
        smdp = SMDPDecisionWrapper(env_factory=env_factory, config=config)
        if args.preseed_buffer:
            print("Dynassign: forcing --preseed-buffer False (no ERB for the dynamic rung yet).")
            args.preseed_buffer = False
        reward_scale = 0.1
    elif args.problem == "hybrid":
        print("Initializing SACRED Stage-2 HYBRID (assignment + next-hop routing, static demand)...")
        from src.envs.assignment_factory import make_hybrid_assign_env

        config = SMDPConfig(
            max_ticks=1500,                 # hybrid routes (next-hop) run longer than destination mode
            antagonist_interval=25,
            congestion_duration=125,        # = 5 x interval -> block expiry aligns to a decision event
            # 1500 = the budget sweep's sweet spot (scratch/critique_probes.py Probe C): scripted
            # route-reach attack costs greedy ~+84% with 8/8 still delivered and episodes ending
            # ~tick 416; 4000 dragged episodes to ~1263 ticks and approaches the everyone-crushed
            # regime. MUST match evaluate_hybrid.hybrid_config.
            congestion_budget=1500.0,
            congestion_cooldown=0,
            congestion_cost=0.1,
            reward_mode="latency",
            routing_mode="hybrid",          # assignment (pick request) + next-hop routing (pick roads)
            routing_corridor_slack=2.0,     # loose enough for a real detour around a blocked gateway
            congestion_levels=(1.0,),       # full-blockage roadblocks
            max_antag_actions_per_event=1,  # one strategic roadblock per decision event
            antag_reach="route",            # block the gateway AHEAD on a truck's route (anticipation)
        )
        smdp = SMDPDecisionWrapper(env_factory=lambda: make_hybrid_assign_env(), config=config)
        if args.preseed_buffer:
            print("Hybrid: forcing --preseed-buffer False.")
            args.preseed_buffer = False
        reward_scale = 0.1
    else:
        print("Initializing SACRED SMDP Kaliningrad OSM Environment...")
        from src.envs.osm_factory import make_osm_env

        config = SMDPConfig(
            max_ticks=600,
            antagonist_interval=30,  # Acts at least every 30 ticks
            congestion_duration=30,
            congestion_budget=500.0,
            congestion_cooldown=0,
            remaining_demand_penalty=0.05,  # protag_reward_shaping: was 0.5 (the dominant, antagonist-driven noise term); demoted to a small urgency nudge
            delivery_reward=100.0,  # protag_reward_shaping: was 10.0; now the dominant, controllable signal so the critic can attribute value to good routing
            time_penalty=1.0,
            congestion_cost=0.1,
            congestion_levels=(0.25, 0.5, 0.75, 1.0)
        )
        smdp = SMDPDecisionWrapper(
            env_factory=lambda: make_osm_env(num_trucks=4, truck_capacity=40.0, episode_packages=150),
            config=config,
        )
        reward_scale = 0.01  # tuned for the OSM baseline's larger per-decision rewards

    # 2. Configure Protagonist SAC Agent
    print("Configuring Protagonist Dispatch Agent...")
    protag_node_dim, protag_edge_dim = 13, 4
    protag_snapshot_sd = None
    if args.protagonist_snapshot is not None:
        import torch
        from src.agents.sac import infer_edge_in_dim, infer_node_in_dim
        protag_snapshot_sd = torch.load(args.protagonist_snapshot, map_location="cpu")
        protag_node_dim = infer_node_in_dim(protag_snapshot_sd)
        protag_edge_dim = infer_edge_in_dim(protag_snapshot_sd)
        print(f"Frozen protagonist from {args.protagonist_snapshot} "
              f"(node_in_dim={protag_node_dim}, edge_in_dim={protag_edge_dim}).")
    protag = ProtagonistSAC(
        node_in_dim=protag_node_dim,
        edge_in_dim=protag_edge_dim,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        heads=4,
        lr_actor=5e-5,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha_init=1.0,
        autotune_alpha=True,
        target_entropy=None,  # None -> sane discrete-SAC fallback; -1.0 caused alpha runaway/critic divergence
        reward_scale=reward_scale,  # problem-dependent (set above): 0.1 stage0 next-hop, 0.01 osm baseline
        device=args.device,
    )

    # 3. Configure Antagonist SAC Agent
    print("Configuring Antagonist Congestion Agent...")
    antag = AntagonistSAC(
        node_in_dim=13,
        edge_in_dim=4,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        heads=4,
        num_congestion_levels=len(config.congestion_levels),
        level_costs=[level * config.congestion_duration for level in config.congestion_levels],
        congestion_levels=config.congestion_levels,
        lr_actor=5e-5,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha_init=1.0,
        autotune_alpha=True,
        target_entropy=None,  # None -> sane discrete-SAC fallback; -1.0 caused alpha runaway/critic divergence
        reward_scale=reward_scale,  # problem-dependent (set above): 0.1 stage0 next-hop, 0.01 osm baseline
        device=args.device,
    )

    if protag_snapshot_sd is not None:
        protag.actor.load_state_dict(protag_snapshot_sd)

    start_episode = 0
    if args.resume_checkpoint is not None:
        import os
        print(f"Resuming from checkpoints in {args.resume_checkpoint}...")
        protag_ckpt = os.path.join(args.resume_checkpoint, "protagonist/checkpoint.pt")
        antag_ckpt = os.path.join(args.resume_checkpoint, "antagonist/checkpoint.pt")
        
        start_episode = protag.load_checkpoint(protag_ckpt)
        antag.load_checkpoint(antag_ckpt)
        print(f"Resumed successfully from episode {start_episode}.")
    elif args.erb_path is not None:
        import os
        import torch
        if not os.path.exists(args.erb_path):
            sys.exit(f"--erb-path {args.erb_path} not found. Generate it first (e.g. generate_erb_assign.py).")
        print(f"\nSeeding protagonist replay buffer from {args.erb_path}...")
        transitions = torch.load(args.erb_path, map_location="cpu", weights_only=False)
        for trans in transitions:
            # Demos are produced by the shared transition builder -> already correct format.
            protag.replay_buffer.push(trans)
        print(f"Seeded {len(transitions)} demonstration transitions.")
    elif args.preseed_buffer:
        import os
        import subprocess
        import torch
        erb_path = "data/erb_transitions.pt"
        if not os.path.exists(erb_path):
            print("\nPre-seeded transition file 'data/erb_transitions.pt' not found.")
            print("Auto-triggering ALNS trajectory generator...")
            # Run generate_erb.py using python
            subprocess.run([sys.executable, "scripts/generate_erb.py"], check=True)
            
        if os.path.exists(erb_path):
            print(f"\nLoading baseline transitions from {erb_path}...")
            transitions = torch.load(erb_path, map_location="cpu", weights_only=False)
            print(f"Pre-seeding protagonist replay buffer with {len(transitions)} transitions...")
            for trans in transitions:
                # Ensure compatibility with GATv2 state structures
                trans.state["allowed_destinations"] = {
                    "protagonist": dict(trans.action_mask["protagonist"])
                }
                protag.replay_buffer.push(trans)
            print("Protagonist replay buffer pre-seeded successfully!")

    # 4. Initialize ATLA Trainer
    print("Initializing ATLA Coevolution Trainer...")
    import datetime
    # Within a generation (--group), name runs by tag+seed (no timestamp) so seeds sit together
    # and are reproducible/identifiable; nesting under <group>/ makes TensorBoard group them and
    # mirrors into models/runs/<group>/. Standalone runs keep the timestamped flat name.
    if args.group is not None:
        seed_suffix = f"_seed{args.seed}" if args.seed is not None else ""
        run_name = f"{args.group}/{args.tag}{seed_suffix}"
    else:
        timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
        run_name = f"{args.tag}_{args.episodes}ep_sw{args.switch_every}_b{args.batch_size}_{timestamp}"
    print(f"Run Name: {run_name}")

    # Periodic learned-vs-greedy eval snapshot (every --eval-every episodes).
    eval_fn = None
    eval_every = 0
    if args.problem == "stage0" and args.eval_every > 0:
        from scripts.evaluate_stage0 import eval_cells
        from src.envs.stage0_factory import make_stage0_nexthop_env as _mk
        eval_fn = lambda ep: eval_cells(protag, antag, lambda: _mk(), config)
        eval_every = args.eval_every
    elif args.problem == "assign" and args.eval_every > 0:
        from scripts.evaluate_assignment import eval_cells_assignment
        from src.envs.assignment_factory import make_assignment_env as _mk
        eval_fn = lambda ep: eval_cells_assignment(protag, antag, lambda: _mk(), config)
        eval_every = args.eval_every
    elif args.problem == "dynassign" and args.eval_every > 0:
        from scripts.evaluate_dynamic_assign import eval_dynamic_cells
        from src.envs.assignment_factory import make_dynamic_assign_env as _mkd
        # Multi-seed, fixed-adversary eval (a few fixed Poisson instances) — the metric the
        # static-3b retraction demands. make_env_for_seed(seed) -> a zero-arg factory bound to it.
        make_env_for_seed = lambda seed: (lambda: _mkd(arrival_rate=args.arrival_rate, demand_seed=seed))
        eval_fn = lambda ep: eval_dynamic_cells(protag, antag, make_env_for_seed, config, seeds=(0, 1, 2))
        eval_every = args.eval_every
    elif args.problem == "hybrid" and args.eval_every > 0:
        from scripts.evaluate_hybrid import eval_hybrid_cells
        from src.envs.assignment_factory import make_hybrid_assign_env as _mkh
        # Static demand -> deterministic single-episode eval vs the current antagonist (progress
        # signal); the real verdict is best-checkpoint vs the FIXED antagonist (select-best, post-hoc).
        eval_fn = lambda ep: eval_hybrid_cells(protag, antag, _mkh, config)
        eval_every = args.eval_every

    trainer = ATLACoevolutionTrainer(
        smdp=smdp,
        protag_agent=protag,
        antag_agent=antag,
        log_dir=args.log_dir,
        switch_every_episodes=args.switch_every,
        batch_size=args.batch_size,
        run_name=run_name,
        eval_fn=eval_fn,
        eval_every=eval_every,
        mode=trainer_mode,
    )

    # 5. Run coevolutionary training
    trainer.run_training(total_episodes=args.episodes, start_episode=start_episode)


if __name__ == "__main__":
    main()
