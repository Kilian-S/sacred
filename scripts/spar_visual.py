#!/usr/bin/env python3
"""Live visualization script to watch the trained GATv2 agents spar off.

Loads the saved coevolved policies from models/ and runs the interactive
PyGame simulator in real-time so you can watch detours and blockages.
"""

from __future__ import annotations

import argparse
from collections import deque
import os
from pathlib import Path
import sys
import torch
import networkx as nx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.graph_env import EdgeId, GraphEnv, NodeId
from src.env.multi_agent import GameConfig, SacredToyGame, describe_game_tick, EpisodeMetrics
from src.env.renderer import PygameToyRenderer
from src.env.toy_graph import make_toy_graph_env
from src.agents.sac import ProtagonistSAC, AntagonistSAC


class TrainedProtagonistPolicy:
    """Dispatcher using the trained GATv2 ProtagonistSAC policy."""

    def __init__(self, agent: ProtagonistSAC, deterministic: bool = False) -> None:
        self.agent = agent
        self.deterministic = deterministic

    def act(self, env: GraphEnv) -> dict[int, NodeId]:
        obs = env.observe()
        
        # 1. Select actions for demonstration sequentially with commitment projection
        import copy
        actions = {}
        projected_obs = copy.deepcopy(obs)
        
        # Gather idle trucks
        idle_truck_ids = [t_id for t_id, t in env.trucks.items() if t.is_idle]
        
        for truck_id in idle_truck_ids:
            truck = env.trucks[truck_id]
            current_node = truck.current_node
            if current_node is None:
                continue
                
            remaining_demand = sum(data["demand"] for _, data in env.graph.nodes(data=True))
            
            if truck.load <= 0 and current_node != env.depot_node:
                allowed = [env.depot_node]
            elif truck.load <= 0:
                allowed = []
            elif remaining_demand <= 0 and current_node != env.depot_node:
                allowed = [env.depot_node]
            elif remaining_demand <= 0:
                allowed = []
            else:
                # Goal-Directed: All customer nodes with positive unassigned demand, plus the depot
                destinations = []
                for n, data in env.graph.nodes(data=True):
                    if not data.get("has_depot", False):
                        node_demand = float(data.get("demand", 0.0))
                        if node_demand > 0.0:
                            # Subtract other trucks' committed capacities using projected_obs!
                            other_targeted = sum(
                                t["load"]
                                for o_id, t in projected_obs["trucks"].items()
                                if o_id != truck_id and t["destination"] == n
                            )
                            unassigned = node_demand - other_targeted
                            if unassigned > 0.0:
                                destinations.append(n)
                
                # Only allow depot if load < capacity, OR if the truck is NOT already at the depot!
                if truck.load < truck.capacity or current_node != env.depot_node:
                    destinations.append(env.depot_node)
                    
                if current_node in destinations and current_node != env.depot_node:
                    destinations.remove(current_node)
                allowed = destinations
                
            if not allowed:
                continue
                
            # Form mask dict for policy evaluation
            truck_mask = {truck_id: allowed}
            
            projected_obs["active_truck"] = truck_id
            projected_obs["allowed_destinations"] = {"protagonist": truck_mask}
            
            action_dict = self.agent.select_action(projected_obs, truck_mask, deterministic=self.deterministic)
            print(f"[DEBUG Protag] Truck {truck_id} at {obs['trucks'][truck_id]['current_node']} | Allowed: {allowed} | Selected: {action_dict}")
            actions.update(action_dict)
            
            # Project commitment
            chosen_node = action_dict.get(truck_id)
            if chosen_node is not None:
                projected_obs["trucks"][truck_id]["destination"] = chosen_node
                projected_obs["trucks"][truck_id]["current_node"] = None
                
        return actions

    def _neighbors_toward(self, env: GraphEnv, source: NodeId, target: NodeId) -> list[NodeId]:
        try:
            path = nx.shortest_path(env.graph, source, target, weight="distance")
        except nx.NetworkXNoPath:
            return []
        if len(path) < 2:
            return []
        return [path[1]]


class TrainedAntagonistPolicy:
    """Congestion adversary using the trained GATv2 AntagonistSAC policy."""

    def __init__(self, agent: AntagonistSAC, interval: int = 60, deterministic: bool = False) -> None:
        self.agent = agent
        self.interval = interval
        self.deterministic = deterministic

    def act(self, env: GraphEnv, game: SacredToyGame) -> dict[EdgeId, float]:
        # Only act at specified decision intervals (matching SMDP)
        if env.time % self.interval != 0 or env.time == 0:
            return {}

        obs = env.observe()
        
        # 1. Compute valid levels by edge under remaining budget constraints
        levels_by_edge = {}
        for u, v in env.graph.edges:
            edge = env._edge_key(u, v)
            if edge in game.active_congestion:
                continue
            valid_levels = [
                level
                for level in [0.25, 0.5, 0.75, 1.0]
                if (level * game.config.congestion_duration) <= game.budget.remaining + 1e-12
            ]
            if valid_levels:
                levels_by_edge[edge] = valid_levels
        
        mask = {
            "can_wait": True,
            "levels_by_edge": levels_by_edge
        }
        
        # 2. Select action
        obs_copy = dict(obs)
        obs_copy["allowed_destinations"] = {
            "antagonist": {
                "allowed_edges": list(levels_by_edge.keys()),
                "original_edges": list(obs["edges"].keys())
            }
        }
        
        action = self.agent.select_action(obs_copy, mask, game.budget.remaining, deterministic=self.deterministic)
        print(f"[DEBUG Antag] Time: {env.time} | Allowed Edges Count: {len(levels_by_edge)} | Selected Action: {action}")
        if action is None:
            return {}
        
        edge, level = action
        return {edge: level}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Watch trained SACRED agents spar in PyGame.")
    parser.add_argument("--ticks", type=int, default=240, help="Maximum episode ticks.")
    parser.add_argument("--fps", type=int, default=30, help="PyGame frames per second.")
    parser.add_argument("--trucks", type=int, default=2, help="Number of trucks.")
    parser.add_argument("--speed", type=float, default=6.0, help="Simulation ticks per second.")
    parser.add_argument("--interval", type=int, default=20, help="Antagonist action interval.")
    parser.add_argument("--budget", type=float, default=120.0, help="Antagonist budget.")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action selection instead of stochastic.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameConfig(
        max_ticks=args.ticks,
        congestion_budget=args.budget,
        congestion_duration=40,
    )

    # 1. Reconstruct agents with correct hidden features
    print("Loading SACRED agent models...")
    protag_agent = ProtagonistSAC(
        node_in_dim=11,
        edge_in_dim=2,
        hidden_dim=64,
        num_layers=2,
        heads=4,
    )
    
    level_costs = [
        level * 40 * 0.015
        for level in [0.25, 0.5, 0.75, 1.0]
    ]
    antag_agent = AntagonistSAC(
        node_in_dim=11,
        edge_in_dim=2,
        hidden_dim=64,
        num_layers=2,
        heads=4,
        num_congestion_levels=4,
        level_costs=level_costs,
    )

    # Load saved weight checkpoints (using absolute paths relative to PROJECT_ROOT)
    protag_path = os.path.join(PROJECT_ROOT, "models", "protagonist", "actor.pt")
    antag_path = os.path.join(PROJECT_ROOT, "models", "antagonist", "actor.pt")
    
    if not (os.path.exists(protag_path) and os.path.exists(antag_path)):
        print(f"Error: Trained policies not found in standard paths!")
        print(f"Looked at absolute paths:")
        print(f"  Protagonist: {protag_path}")
        print(f"  Antagonist:  {antag_path}")
        sys.exit(1)

    protag_agent.actor.load_state_dict(torch.load(protag_path, map_location="cpu"))
    antag_agent.actor.load_state_dict(torch.load(antag_path, map_location="cpu"))
    print("Agent policies loaded successfully!")

    # 2. Instantiate custom policies and register them in the game loop
    protagonist = TrainedProtagonistPolicy(protag_agent, deterministic=args.deterministic)
    antagonist = TrainedAntagonistPolicy(antag_agent, interval=args.interval, deterministic=args.deterministic)

    game = SacredToyGame(
        env_factory=lambda: make_toy_graph_env(num_trucks=args.trucks, max_time=args.ticks),
        protagonist=protagonist,
        antagonist=antagonist,
        config=config,
    )
    game.reset()

    # 3. Open PyGame and run simulation
    print("\nStarting PyGame live visualization...")
    print("Use SPACEBAR to pause/resume the simulation.")
    renderer = PygameToyRenderer(fps=args.fps, sim_ticks_per_second=args.speed)
    
    try:
        running = True
        latest = None
        feed = deque(maxlen=80)
        while running and not game.env.is_done() and game.metrics.ticks < config.max_ticks:
            if renderer.should_advance():
                latest = game.step()
                feed.extend(describe_game_tick(latest))
            running = renderer.render(
                game.env,
                latest.metrics if latest is not None else game.metrics,
                protagonist_reward=latest.protagonist_reward if latest is not None else 0.0,
                antagonist_reward=latest.antagonist_reward if latest is not None else 0.0,
                antagonist_action=latest.antagonist_action if latest is not None else {},
                feed=list(feed),
            )
    finally:
        renderer.close()
        print("\nVisualization closed successfully!")


if __name__ == "__main__":
    main()
