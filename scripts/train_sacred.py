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
    args = parser.parse_args()

    # 1. Initialize SMDP Environment with Kaliningrad OSM Graph (4 trucks)
    print("Initializing SACRED SMDP Kaliningrad OSM Environment...")
    from src.envs.osm_factory import make_osm_env
    
    config = SMDPConfig(
        max_ticks=600,
        antagonist_interval=30,  # Acts at least every 30 ticks
        congestion_duration=30,
        congestion_budget=500.0,
        congestion_cooldown=0,
        remaining_demand_penalty=0.5,
        delivery_reward=10.0,
        time_penalty=1.0,
        congestion_cost=0.1,
        congestion_levels=(0.25, 0.5, 0.75, 1.0)
    )
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_osm_env(num_trucks=4, truck_capacity=40.0, episode_packages=150),
        config=config,
    )

    # 2. Configure Protagonist SAC Agent
    print("Configuring Protagonist Dispatch Agent...")
    protag = ProtagonistSAC(
        node_in_dim=9,
        edge_in_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        heads=4,
        lr_actor=5e-5,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha_init=1.0,
        autotune_alpha=True,
        target_entropy=-1.0,
        device=args.device,
    )

    # 3. Configure Antagonist SAC Agent
    print("Configuring Antagonist Congestion Agent...")
    antag = AntagonistSAC(
        node_in_dim=9,
        edge_in_dim=2,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        heads=4,
        num_congestion_levels=len(config.congestion_levels),
        level_costs=[level * config.congestion_duration for level in config.congestion_levels],
        lr_actor=5e-5,
        lr_critic=1e-3,
        gamma=0.99,
        tau=0.005,
        alpha_init=1.0,
        autotune_alpha=True,
        target_entropy=-1.0,
        device=args.device,
    )

    start_episode = 0
    if args.resume_checkpoint is not None:
        import os
        print(f"Resuming from checkpoints in {args.resume_checkpoint}...")
        protag_ckpt = os.path.join(args.resume_checkpoint, "protagonist/checkpoint.pt")
        antag_ckpt = os.path.join(args.resume_checkpoint, "antagonist/checkpoint.pt")
        
        start_episode = protag.load_checkpoint(protag_ckpt)
        antag.load_checkpoint(antag_ckpt)
        print(f"Resumed successfully from episode {start_episode}.")
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
    timestamp = datetime.datetime.now().strftime("%m%d_%H%M%S")
    run_name = f"{args.tag}_{args.episodes}ep_sw{args.switch_every}_b{args.batch_size}_{timestamp}"
    print(f"Run Name: {run_name}")

    trainer = ATLACoevolutionTrainer(
        smdp=smdp,
        protag_agent=protag,
        antag_agent=antag,
        log_dir=args.log_dir,
        switch_every_episodes=args.switch_every,
        batch_size=args.batch_size,
        run_name=run_name,
    )

    # 5. Run coevolutionary training
    trainer.run_training(total_episodes=args.episodes, start_episode=start_episode)


if __name__ == "__main__":
    main()
