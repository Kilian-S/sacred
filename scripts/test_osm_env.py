from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.multi_agent import GameConfig, SacredToyGame
from src.envs.osm_factory import make_osm_env
from src.env.renderer import PygameToyRenderer
from collections import deque

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("1. Initializing the Kaliningrad OSM Environment...")
    config = GameConfig(
        max_ticks=600,
        congestion_budget=500.0,
        congestion_level=1.0,  # Will be mapped to bins shortly
        congestion_duration=30,
        congestion_cooldown=0,
    )
    
    game = SacredToyGame(
        env_factory=lambda: make_osm_env(num_trucks=4, truck_capacity=40.0, episode_packages=150),
        config=config,
    )
    game.reset()
    
    print(f"-> Environment ready! Nodes: {len(game.env.graph.nodes)}, Edges: {len(game.env.graph.edges)}")
    print("2. Launching 4 Trucks...")

    if args.headless:
        print("Running Headless 50-tick test...")
        for _ in range(50):
            game.step()
        print(f"Finished 50 ticks! Delivered: {game.metrics.total_delivery}")
        return

    # If not headless, boot up the dynamic PyGame renderer!
    renderer = PygameToyRenderer(width=1400, height=900, fps=60, sim_ticks_per_second=20.0)
    
    running = True
    latest = None
    while running and not game.env.is_done() and game.metrics.ticks < config.max_ticks:
        if renderer.should_advance():
            latest = game.step()
            
        running = renderer.render(
            game.env,
            latest.metrics if latest is not None else game.metrics,
            protagonist_reward=latest.protagonist_reward if latest is not None else 0.0,
            antagonist_reward=latest.antagonist_reward if latest is not None else 0.0,
            antagonist_action=latest.antagonist_action if latest is not None else {}
        )

if __name__ == '__main__':
    main()
