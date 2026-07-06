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
             "buffer (e.g. data/erb_assign.pt from generate_erb_assign.py); demos are produced by "
             "the shared transition builder (no compat shim needed).",
    )
    parser.add_argument(
        "--problem",
        type=str,
        choices=["osm", "stage0", "assign", "dynassign", "hybrid", "contested"],
        default="osm",
        help=(
            "Which problem to train. 'osm' (default) = the static-demand Kaliningrad "
            "baseline. 'stage0' = single-truck next-hop route-choice validation rung. "
            "'assign' = the 3b multi-truck assignment probe (static, RETIRED baseline). "
            "'dynassign' = Stage 1.5 dynamic assignment (Poisson arrivals, 2 trucks, latency). "
            "'hybrid' = Stage 2 hybrid (assignment + next-hop routing, chokepoint geometry, static). "
            "'contested' = gen07 contested-resupply arena (dynassign dynamics + route-reach "
            "antagonist; the exploitability headline arena — see src/envs/contested.py)."
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
    parser.add_argument(
        "--update-every",
        type=int,
        default=1,
        help="Run a gradient update once every N decision epochs (per agent) instead of every one. "
             "1 = historical behaviour. The hybrid rung makes ~10x more (edge-level) decisions per "
             "episode than destination-mode rungs, where update-per-decision is prohibitive. Use "
             "the SAME value for every arm of a generation.",
    )
    parser.add_argument(
        "--reward-baseline",
        type=str,
        choices=["none", "twin"],
        default="none",
        help="B1 counterfactual reward baseline (contested arena). 'none' (default) = the raw "
             "latency reward, unchanged for every historical run. 'twin' = subtract a per-tick "
             "greedy no-attack baseline replaying the same arrivals (difference reward: strips the "
             "arrival trend + unavoidable-under-clean-greedy damage that floods the signal under "
             "attack, the gen06 M1 SNR fix). Zero-sum preserved up to a per-episode constant.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="SAC discount factor (applied per SMDP tick as gamma^elapsed_ticks). Default 0.99 "
             "reproduces every historical run. gen07 (B5) raises it to ~0.997 to fight the "
             "gamma-myopia diagnosed in gen04 (0.99^300~0.05 buries the queue-compounding payoff; "
             "0.997^300~0.41 keeps it in horizon). Applied to BOTH agents; use the SAME value for "
             "every arm of a generation.",
    )
    parser.add_argument(
        "--protag-target-entropy",
        type=float,
        default=None,
        help="B2 entropy repair. None (default) = the historical dynamic target 0.45*ln(N) over "
             "the action-set size N. Setting an ABSOLUTE value decouples the target from N, which "
             "under attack is the inflated pending-queue size (the M2 pathology: the attack forces "
             "the max-entropy objective to demand MORE randomness exactly where indecision is "
             "punished). Protagonist only; use the SAME value across a generation's arms.",
    )
    parser.add_argument(
        "--antag-target-entropy",
        type=float,
        default=None,
        help="B2 entropy repair, antagonist side (the gen04b hypothesis made testable). None "
             "(default) = the historical 0.5*ln(N) target that pinned the attacker near-uniform "
             "(gen04 entropy pinning). A LOWER absolute value lets the attacker commit to a "
             "strategic block instead of being mandated toward the random attacker.",
    )
    parser.add_argument(
        "--scripted-attacker",
        type=str,
        choices=["targeted", "pathrand"],
        default="targeted",
        help="Which scripted attacker --scripted-adversary trains against. gen05 hybrid used "
             "'targeted'; gen06 dynassign trains vs 'pathrand' (random goal-committed truck's "
             "path edge) so 'targeted' stays HELD OUT as the test attack.",
    )
    parser.add_argument(
        "--scripted-adversary",
        action="store_true",
        help="Adversarial training against the FIXED scripted targeted attacker (gen04 consequence:"
             " the learned adversary cannot learn to attack). The protagonist trains every episode;"
             " the antagonist nets are never used.",
    )
    parser.add_argument(
        "--attack-curriculum",
        action="store_true",
        help="B3: ramp attack exposure/strength instead of the gen06 constant-full-budget regime "
             "(scripted_adversary only). Mixes clean/attacked episodes (--p-attack) and raises the "
             "budget from --budget-min toward the config budget only while windowed delivery stays "
             ">= --competence-floor. Addresses the M3 collapse-regime training distribution.",
    )
    parser.add_argument("--p-attack", type=float, default=0.75,
                        help="B3: probability an episode is attacked (default 0.75).")
    parser.add_argument("--budget-min", type=float, default=500.0,
                        help="B3: attack budget at curriculum level 0 (ramps to the config budget).")
    parser.add_argument("--curriculum-levels", type=int, default=4,
                        help="B3: number of budget ramp levels (default 4).")
    parser.add_argument("--competence-floor", type=float, default=0.4,
                        help="B3: windowed delivery rate required (over attacked episodes) to ramp "
                             "difficulty up one level (default 0.4).")
    parser.add_argument("--curriculum-window", type=int, default=20,
                        help="B3: number of attacked episodes averaged for the ramp gate (default 20).")
    parser.add_argument(
        "--attacker-mixture",
        type=str,
        default=None,
        help="B4-lite adversary population (scripted_adversary only). Comma spec of "
             "name:weight, e.g. 'targeted:1,pathrand:1,gateway:1'. One member is sampled per "
             "episode (fictitious-play-flavoured; denies the policy a single attacker to overfit). "
             "Overrides --scripted-attacker. Names: targeted, pathrand, gateway, random.",
    )
    args = parser.parse_args()

    if sum([args.vanilla, args.train_antagonist_only, args.scripted_adversary]) > 1:
        sys.exit("--vanilla, --train-antagonist-only and --scripted-adversary are mutually exclusive.")
    if args.train_antagonist_only and not args.protagonist_snapshot:
        sys.exit("--train-antagonist-only requires --protagonist-snapshot.")
    if args.vanilla:
        trainer_mode = "vanilla"
    elif args.train_antagonist_only:
        trainer_mode = "antagonist_only"
    elif args.scripted_adversary:
        trainer_mode = "scripted_adversary"
    else:
        trainer_mode = "atla"

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
        reward_scale = 0.1
    elif args.problem == "contested":
        print("Initializing SACRED contested-resupply arena (dynassign dynamics + route reach)...")
        import dataclasses
        import itertools
        from src.envs.contested import (
            contested_config, make_contested_env, make_greedy_twin_baseline_provider)

        # Single source of truth for the arena config (train + eval share it; see contested.py).
        config = contested_config(congestion_budget=args.congestion_budget)
        # B1: opt into the twin difference reward on the config + inject the provider (default off).
        baseline_provider = None
        if args.reward_baseline == "twin":
            config = dataclasses.replace(config, reward_baseline="twin")
            baseline_provider = make_greedy_twin_baseline_provider(config, arrival_rate=args.arrival_rate)
            print("Reward baseline: greedy no-attack twin (B1 difference reward).")
        # Same reproducible per-episode Poisson demand stream as dynassign (fresh env each reset;
        # a counter advances the stream so a fixed seed does not repeat one instance).
        if args.seed is not None:
            _demand_counter = itertools.count(args.seed * 100003)
            env_factory = lambda: make_contested_env(
                arrival_rate=args.arrival_rate, demand_seed=next(_demand_counter))
        else:
            env_factory = lambda: make_contested_env(arrival_rate=args.arrival_rate)
        smdp = SMDPDecisionWrapper(env_factory=env_factory, config=config, baseline_provider=baseline_provider)
        reward_scale = 0.1
    elif args.problem == "hybrid":
        print("Initializing SACRED Stage-2 HYBRID (assignment + next-hop routing, static demand)...")
        from src.envs.assignment_factory import make_hybrid_assign_env

        config = SMDPConfig(
            max_ticks=800,                  # greedy ends ~tick 220 unattacked / ~416 under the
                                            # budget-1500 attack (post zombie-fix); 800 keeps full
                                            # headroom while halving the cost of untrained
                                            # wandering. MUST match evaluate_hybrid.hybrid_config.
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
        gamma=args.gamma,
        tau=0.005,
        alpha_init=1.0,
        autotune_alpha=True,
        # B2: absolute target if --protag-target-entropy set, else the historical dynamic
        # 0.45*ln(N) fallback (None). -1.0 caused alpha runaway/critic divergence — never that.
        target_entropy=args.protag_target_entropy,
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
        gamma=args.gamma,
        tau=0.005,
        alpha_init=1.0,
        autotune_alpha=True,
        # B2: absolute target if --antag-target-entropy set (the gen04b entropy-pinning fix),
        # else the historical dynamic 0.5*ln(N) fallback (None).
        target_entropy=args.antag_target_entropy,
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
    elif args.problem == "contested" and args.eval_every > 0:
        from scripts.evaluate_dynamic_assign import eval_dynamic_cells
        from src.envs.contested import make_contested_env as _mkc
        # Same dynamics as dynassign -> reuse its multi-seed fixed-adversary eval, with the
        # contested (route-reach) config so the periodic signal matches training.
        make_env_for_seed = lambda seed: (lambda: _mkc(arrival_rate=args.arrival_rate, demand_seed=seed))
        eval_fn = lambda ep: eval_dynamic_cells(protag, antag, make_env_for_seed, config, seeds=(0, 1, 2))
        eval_every = args.eval_every
    elif args.problem == "hybrid" and args.eval_every > 0:
        from scripts.evaluate_hybrid import eval_hybrid_cells
        from src.envs.assignment_factory import make_hybrid_assign_env as _mkh
        # Static demand -> deterministic single-episode eval vs the current antagonist (progress
        # signal); the real verdict is best-checkpoint vs the FIXED antagonist (select-best, post-hoc).
        eval_fn = lambda ep: eval_hybrid_cells(protag, antag, _mkh, config)
        eval_every = args.eval_every

    scripted_attacker = None
    scripted_attacker_pool = None
    if trainer_mode == "scripted_adversary":
        from src.baselines.attackers import (
            ScriptedAttackerMixture, build_scripted_attacker,
            random_path_block_policy, targeted_block_policy)
        if args.attacker_mixture:
            # B4-lite: parse "name:weight,..." into a per-episode-sampled population.
            members = []
            for i, part in enumerate(args.attacker_mixture.split(",")):
                name, _, w = part.strip().partition(":")
                weight = float(w) if w else 1.0
                # Distinct seed per stochastic member so mixture draws don't correlate.
                members.append((name, build_scripted_attacker(name, smdp, seed=(args.seed or 0) + i), weight))
            scripted_attacker_pool = ScriptedAttackerMixture(members, seed=args.seed or 0)
            scripted_attacker = scripted_attacker_pool.sample()[1]  # placeholder; resampled per episode
            print(f"Scripted adversary POPULATION (B4-lite): {args.attacker_mixture}")
        else:
            if args.scripted_attacker == "pathrand":
                scripted_attacker = random_path_block_policy(smdp, seed=args.seed or 0)
            else:
                scripted_attacker = targeted_block_policy(smdp)
            print(f"Scripted adversary: {args.scripted_attacker}")

    attack_curriculum = None
    if args.attack_curriculum:
        if trainer_mode != "scripted_adversary":
            sys.exit("--attack-curriculum requires --scripted-adversary.")
        from src.agents.curriculum import AttackCurriculum
        attack_curriculum = AttackCurriculum(
            budget_min=args.budget_min,
            budget_max=config.congestion_budget,
            n_levels=args.curriculum_levels,
            p_attack=args.p_attack,
            competence_floor=args.competence_floor,
            window=args.curriculum_window,
            seed=args.seed or 0,
        )
        print(f"Attack curriculum: p_attack={args.p_attack}, budget "
              f"{args.budget_min}->{config.congestion_budget} over {args.curriculum_levels} levels, "
              f"floor {args.competence_floor}, window {args.curriculum_window}")

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
        scripted_attacker=scripted_attacker,
        update_every=args.update_every,
        attack_curriculum=attack_curriculum,
        scripted_attacker_pool=scripted_attacker_pool,
    )

    # 5. Run coevolutionary training
    trainer.run_training(total_episodes=args.episodes, start_episode=start_episode)


if __name__ == "__main__":
    main()
