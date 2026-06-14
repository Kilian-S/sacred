#!/usr/bin/env python3
"""Run Adaptive Large Neighborhood Search (ALNS) and save data to data/erb_transitions.pt."""

from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.smdp_wrapper import SMDPDecisionWrapper, SMDPConfig, DecisionType, SMDPTransition
from src.env.toy_graph import make_toy_graph_env
from src.baselines.metaheuristic import AdaptiveLargeNeighborhoodSearchVRP


def main() -> None:
    print("=" * 60)
    print("     SACRED PHASE 4: EXPERIENCE REPLAY BOOTSTRAPPING (ERB)    ")
    print("=" * 60)

    # 1. Initialize SMDP Environment with Toy Graph (2 trucks)
    print("Initializing environment...")
    config = SMDPConfig(
        max_ticks=240,
        antagonist_interval=20,
        congestion_duration=40,
        congestion_budget=120.0,
        remaining_demand_penalty=0.08,
    )
    # Create the raw environment first to solve it statically
    raw_env = make_toy_graph_env(num_trucks=2)
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: make_toy_graph_env(num_trucks=2),
        config=config,
    )

    # 2. Run ALNS VRP Solver on initial demand configuration
    print("Solving static VRP layout using ALNS...")
    alns = AdaptiveLargeNeighborhoodSearchVRP(raw_env, iterations=500)
    best_sol = alns.solve()

    print("\nOptimal ALNS Routing Plan Found:")
    for t_id, seq in best_sol.items():
        nodes_seq = [t[0] for t in seq]
        print(f"  Truck {t_id} customer visits: {nodes_seq}")

    # Reconstruct the high-level destination sequences
    truck_paths = {}
    for t_id in best_sol:
        truck_paths[t_id] = alns.get_high_level_destinations(best_sol[t_id])
        print(f"  Truck {t_id} high-level destinations: {truck_paths[t_id]}")

    # Track path indexes for each truck
    path_indices = {t_id: 0 for t_id in best_sol}  # Start index at 0 since depot is not explicitly in best_sol VRP solution format but get_high_level_destinations already starts with first target/depot

    # 3. Simulate environment and capture SMDP transitions
    print("\nSimulating ALNS routes in environment to gather trajectories...")
    event = smdp.reset_decision_env()
    transitions: list[SMDPTransition] = []
    
    ep_ticks = 0
    ep_reward = 0.0

    while not event.done:
        dt = event.elapsed_ticks
        ep_ticks += dt

        # A. PROTAGONIST DECISION POINT
        if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            mask = event.protagonist_action_mask
            
            # Follow ALNS path sequentially with state projection
            import copy
            actions = {}
            projected_obs = copy.deepcopy(event.observation)
            truck_decision_states = {}

            for truck_id in event.waiting_trucks:
                projected_obs["active_truck"] = truck_id
                projected_obs["allowed_destinations"] = {"protagonist": dict(mask)}
                
                # Capture exact projected state this truck sees
                truck_decision_states[truck_id] = copy.deepcopy(projected_obs)

                path = truck_paths[truck_id]
                idx = path_indices[truck_id]
                
                if idx < len(path):
                    next_node = path[idx]
                    path_indices[truck_id] += 1
                else:
                    next_node = smdp.env.depot_node

                actions[truck_id] = next_node
                
                # Project commitment
                projected_obs["trucks"][truck_id]["destination"] = next_node
                projected_obs["trucks"][truck_id]["current_node"] = None

            # Step environment
            next_event, transition = smdp.step_protagonist(actions)

            # Generate individual truck-level transitions for ReplayBuffer compatibility
            for truck_id in event.waiting_trucks:
                state_copy = truck_decision_states[truck_id]

                next_state_copy = dict(next_event.observation)
                if next_event.waiting_trucks:
                    next_state_copy["active_truck"] = next_event.waiting_trucks[0]
                else:
                    next_state_copy["active_truck"] = None
                next_state_copy["allowed_destinations"] = {
                    "protagonist": dict(next_event.protagonist_action_mask)
                }

                # Construct clean transition record
                t_trans = SMDPTransition(
                    agent="protagonist",
                    state=state_copy,
                    action=actions,
                    reward=transition.reward,
                    next_state=next_state_copy,
                    done=transition.done,
                    elapsed_ticks=transition.elapsed_ticks,
                    action_mask={"protagonist": dict(mask)},
                    info=dict(transition.info)
                )
                transitions.append(t_trans)

            ep_reward += transition.reward
            event = next_event

        # B. ANTAGONIST DECISION POINT (Choose No-Op/Wait in static demonstration)
        elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
            # Adversary chooses wait action (None)
            next_event, transition = smdp.step_antagonist(None)
            ep_reward += next_event.protagonist_reward
            event = next_event

        else:
            event = smdp.advance_until_decision()

    print(f"\nSimulation complete: Ticks: {ep_ticks} | Total Protag Reward: {ep_reward:.2f}")

    # 4. Serialize transitions to disk
    os.makedirs("data", exist_ok=True)
    save_path = "data/erb_transitions.pt"
    torch.save(transitions, save_path)
    print(f"Successfully serialized {len(transitions)} transitions to {save_path}!")


if __name__ == "__main__":
    main()
