"""Launch the early SACRED toy demo.

Run live visualization:
    .venv/bin/python demo_sacred.py

Run headlessly:
    .venv/bin/python demo_sacred.py --headless

Simulation semantics:
    1 environment tick = 1 simulated second.
"""

from __future__ import annotations

import argparse
from collections import deque

from src.env.multi_agent import GameConfig, NoOpAntagonist, SacredToyGame, describe_game_tick
from src.env.renderer import PygameToyRenderer
from src.env.toy_graph import make_toy_graph_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the early SACRED toy demo.")
    parser.add_argument("--headless", action="store_true", help="Run without opening a PyGame window.")
    parser.add_argument("--no-antagonist", action="store_true", help="Disable the adversarial congestion agent.")
    parser.add_argument("--ticks", type=int, default=240, help="Maximum episode length.")
    parser.add_argument("--trucks", type=int, default=2, help="Number of trucks in the toy fleet.")
    parser.add_argument("--fps", type=int, default=30, help="Visualization frame rate.")
    parser.add_argument("--speed", type=float, default=6.0, help="Initial simulation ticks per second.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GameConfig(max_ticks=args.ticks)
    antagonist = NoOpAntagonist() if args.no_antagonist else None
    game = SacredToyGame(
        env_factory=lambda: make_toy_graph_env(num_trucks=args.trucks, max_time=args.ticks),
        antagonist=antagonist,
        config=config,
    )
    game.reset()

    if args.headless:
        print_summary(game.run_episode())
        return

    renderer = PygameToyRenderer(fps=args.fps, sim_ticks_per_second=args.speed)
    latest = None
    feed = deque(maxlen=80)
    try:
        running = True
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

    print_summary(latest.metrics if latest is not None else game.metrics)


def print_summary(metrics) -> None:
    print("SACRED early demo complete")
    print(f"ticks={metrics.ticks}")
    print(f"done_reason={metrics.done_reason}")
    print(f"delivered={metrics.total_delivery:.0f}")
    print(f"distance={metrics.total_distance:.2f}")
    print(f"protagonist_return={metrics.protagonist_return:.3f}")
    print(f"antagonist_return={metrics.antagonist_return:.3f}")
    print(f"congestion_budget_used={metrics.congestion_budget_used:.2f}")
    print(f"congestion_events={metrics.congestion_events}")


if __name__ == "__main__":
    main()
