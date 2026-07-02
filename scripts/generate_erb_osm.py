#!/usr/bin/env python3
"""Generate an ERB (Experience Replay Bootstrap) dataset on the OSM training env.

Each episode: build ONE OSM env (a fresh random demand instance), solve it with ALNS,
then replay that optimal static plan through the SAME env (adversary idle) and capture the
protagonist's decision transitions. Building/solving/simulating one env keeps the demand
consistent (GraphEnv.reset restores `_initial_graph`), so the demonstrations are valid.

Episodes are independent, so they run across worker processes. The reward config MUST mirror
scripts/train_sacred.py so the demo rewards match the training reward (keep in sync).

Usage:
    PYTHONPATH=. python scripts/generate_erb_osm.py --episodes 50 --iterations 150 --workers 8
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reward config mirrored from scripts/train_sacred.py (protag_reward_shaping). Keep in sync.
CONFIG_KWARGS = dict(
    max_ticks=600,
    antagonist_interval=30,
    congestion_duration=30,
    congestion_budget=500.0,
    congestion_cooldown=0,
    remaining_demand_penalty=0.05,
    delivery_reward=100.0,
    time_penalty=1.0,
    congestion_cost=0.1,
    congestion_levels=(0.25, 0.5, 0.75, 1.0),
)


def _generate_episode(args: tuple[int, int]) -> list:
    """Worker: one ALNS-demonstrated episode on a fresh OSM demand instance."""
    ep_idx, iterations = args
    from src.env.smdp_wrapper import SMDPDecisionWrapper, SMDPConfig, DecisionType, SMDPTransition
    from src.envs.osm_factory import make_osm_env
    from src.baselines.metaheuristic import AdaptiveLargeNeighborhoodSearchVRP

    config = SMDPConfig(**CONFIG_KWARGS)
    env = make_osm_env(num_trucks=4, truck_capacity=40.0, episode_packages=150)
    initial_demand = env.remaining_demand

    # Solve the static VRP for THIS demand instance.
    alns = AdaptiveLargeNeighborhoodSearchVRP(env, iterations=iterations)
    best_sol = alns.solve()
    truck_paths = {t_id: alns.get_high_level_destinations(best_sol[t_id]) for t_id in best_sol}
    path_indices = {t_id: 0 for t_id in best_sol}

    # Replay the plan through the SAME env (reset restores the demand ALNS solved on).
    smdp = SMDPDecisionWrapper(env_factory=lambda: env, config=config)
    event = smdp.reset_decision_env()
    assert abs(smdp.env.remaining_demand - initial_demand) < 1e-6, "demand coupling broken"

    transitions: list = []
    while not event.done:
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            import copy
            mask = event.protagonist_action_mask
            actions = {}
            projected_obs = copy.deepcopy(event.observation)
            truck_decision_states = {}
            for truck_id in event.waiting_trucks:
                projected_obs["active_truck"] = truck_id
                projected_obs["allowed_destinations"] = {"protagonist": dict(mask)}
                truck_decision_states[truck_id] = copy.deepcopy(projected_obs)
                path = truck_paths[truck_id]
                idx = path_indices[truck_id]
                if idx < len(path):
                    next_node = path[idx]
                    path_indices[truck_id] += 1
                else:
                    next_node = smdp.env.depot_node
                actions[truck_id] = next_node
                projected_obs["trucks"][truck_id]["destination"] = next_node
                projected_obs["trucks"][truck_id]["current_node"] = None

            next_event, transition = smdp.step_protagonist(actions)
            for truck_id in event.waiting_trucks:
                state_copy = truck_decision_states[truck_id]
                next_state_copy = dict(next_event.observation)
                next_state_copy["active_truck"] = (
                    next_event.waiting_trucks[0] if next_event.waiting_trucks else None
                )
                next_state_copy["allowed_destinations"] = {
                    "protagonist": dict(next_event.protagonist_action_mask)
                }
                transitions.append(SMDPTransition(
                    agent="protagonist",
                    state=state_copy,
                    action=actions,
                    reward=transition.reward,
                    next_state=next_state_copy,
                    done=transition.done,
                    elapsed_ticks=transition.elapsed_ticks,
                    action_mask={"protagonist": dict(mask)},
                    info=dict(transition.info),
                ))
            event = next_event

        elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
            next_event, _ = smdp.step_antagonist(None)  # adversary idle in demonstrations
            event = next_event
        else:
            event = smdp.advance_until_decision()

    return transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an OSM ERB dataset via parallel ALNS.")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=150, help="ALNS iterations per solve (convergence ~plateaus by ~150 on this graph)")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1),
                        help="parallel ALNS workers (default: cpu_count-1, e.g. 9 on a 10-core M4; use --workers 10 to max)")
    parser.add_argument("--out", type=str, default="data/erb_transitions_osm.pt")
    args = parser.parse_args()

    print(f"Generating {args.episodes} ALNS episodes on OSM "
          f"({args.iterations} iters/solve, {args.workers} workers)...")
    t0 = time.perf_counter()
    work = [(i, args.iterations) for i in range(args.episodes)]
    transitions: list = []
    with Pool(processes=args.workers) as pool:
        for k, ep_trans in enumerate(pool.imap_unordered(_generate_episode, work), 1):
            transitions.extend(ep_trans)
            print(f"  [{k}/{args.episodes}] +{len(ep_trans)} transitions "
                  f"(total {len(transitions)}, {time.perf_counter()-t0:.0f}s)", flush=True)

    os.makedirs("data", exist_ok=True)
    torch.save(transitions, args.out)
    print(f"\nSaved {len(transitions)} transitions to {args.out} in {time.perf_counter()-t0:.0f}s "
          f"({len(transitions)/max(1,args.episodes):.0f}/episode).")


if __name__ == "__main__":
    main()
