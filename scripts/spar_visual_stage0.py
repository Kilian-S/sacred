#!/usr/bin/env python3
"""Live PyGame visualization of the Stage-0 NEXT-HOP agents sparring on the OSM corridor.

Unlike scripts/spar_visual.py (toy graph, destination-mode), this drives the *current*
setup: `make_stage0_nexthop_env` through `SMDPDecisionWrapper` with `routing_mode="next_hop"`,
so you watch the protagonist choose each edge (fast vs safe route around the depot 14 ->
target 82 corridor) while the antagonist injects congestion. It renders every simulated
tick (smooth truck movement) and reuses the wrapper's masks/budget so behaviour matches
training/eval exactly.

Usage:
    PYTHONPATH=. python scripts/spar_visual_stage0.py --run models/runs/<run_name>
    PYTHONPATH=. python scripts/spar_visual_stage0.py            # uses models/{protagonist,antagonist}/actor.pt
Keys: SPACE pause/resume, RIGHT step, UP/DOWN speed, ESC quit.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.smdp_wrapper import SMDPConfig, SMDPDecisionWrapper
from src.env.multi_agent import EpisodeMetrics
from src.env.renderer import PygameToyRenderer
from src.envs.stage0_factory import make_stage0_nexthop_env
from src.agents.sac import AntagonistSAC, ProtagonistSAC


def stage0_config() -> SMDPConfig:
    """Must match scripts/train_sacred.py's stage0 branch."""
    return SMDPConfig(
        max_ticks=400, antagonist_interval=20, congestion_duration=30,
        congestion_budget=300.0, congestion_cooldown=0, congestion_cost=0.1,
        reward_mode="latency", routing_mode="next_hop", congestion_levels=(0.25, 0.5, 0.75, 1.0),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Watch Stage-0 next-hop agents spar.")
    p.add_argument("--run", default=None, help="models/runs/<name> dir with protagonist/actor.pt + antagonist/actor.pt")
    p.add_argument("--speed", type=float, default=8.0, help="Simulation ticks per second.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--deterministic", action="store_true", help="Greedy (argmax) action selection.")
    return p.parse_args()


def load_agents(run: str | None, cfg: SMDPConfig):
    protag = ProtagonistSAC(node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4, device="cpu")
    antag = AntagonistSAC(
        node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(cfg.congestion_levels),
        level_costs=[lvl * cfg.congestion_duration for lvl in cfg.congestion_levels], device="cpu",
    )
    if run:
        pp = os.path.join(run, "protagonist", "actor.pt")
        ap = os.path.join(run, "antagonist", "actor.pt")
    else:
        pp = os.path.join(PROJECT_ROOT, "models", "protagonist", "actor.pt")
        ap = os.path.join(PROJECT_ROOT, "models", "antagonist", "actor.pt")
    if not (os.path.exists(pp) and os.path.exists(ap)):
        sys.exit(f"Checkpoints not found:\n  {pp}\n  {ap}\nPass --run models/runs/<name>.")
    protag.actor.load_state_dict(torch.load(pp, map_location="cpu"))
    antag.actor.load_state_dict(torch.load(ap, map_location="cpu"))
    print(f"Loaded protagonist: {pp}\nLoaded antagonist:  {ap}")
    return protag, antag


def main() -> None:
    args = parse_args()
    cfg = stage0_config()
    protag, antag = load_agents(args.run, cfg)

    smdp = SMDPDecisionWrapper(env_factory=lambda: make_stage0_nexthop_env(), config=cfg)
    smdp.reset_decision_env()
    env = smdp.env
    det = args.deterministic
    print(f"Corridor: depot {env.stage0_depot} -> target {env.stage0_target} (demand {env.remaining_demand:.0f})")

    metrics = EpisodeMetrics()
    initial_demand = env.remaining_demand
    last_cong: dict = {}

    def one_tick() -> None:
        nonlocal last_cong
        # 1. Protagonist next-hop decisions for idle trucks (forward/branch mask).
        next_hop = {}
        mask = smdp.protagonist_action_mask()
        for tid, opts in mask.items():
            if not opts:
                continue
            if len(opts) >= 2:  # genuine route choice -> ask the policy
                obs = env.observe()
                obs["active_truck"] = tid
                obs["allowed_destinations"] = {"protagonist": dict(mask)}
                next_hop.update(protag.select_action(obs, mask, deterministic=det))
            else:  # forced move
                next_hop[tid] = opts[0]

        # 2. Antagonist acts on its interval (reuse wrapper masking + budget accounting).
        cong: dict = {}
        if env.time > 0 and env.time % cfg.antagonist_interval == 0:
            amask = smdp.antagonist_action_mask()
            if any(amask.get("levels_by_edge", {}).values()):
                obs = env.observe()
                obs["allowed_destinations"] = {"antagonist": {
                    "allowed_edges": list(amask["levels_by_edge"].keys()),
                    "original_edges": list(obs["edges"].keys()),
                }}
                action = antag.select_action(obs, amask, smdp.budget.remaining, deterministic=det)
                applied = smdp._valid_antagonist_action(action, amask)  # spends budget, registers expiry
                for edge, level in applied.items():
                    env.set_congestion(edge, level)
                    cong[edge] = level
                    metrics.congestion_events += 1
        if cong:
            last_cong = cong

        # 3. Advance one simulated second, then age congestion (mirrors the wrapper's tick order).
        result = env.step(next_hop_dispatch=next_hop)
        smdp._age_congestion()

        # 4. Panel metrics.
        metrics.ticks = env.time
        metrics.total_delivery = initial_demand - env.remaining_demand
        metrics.total_distance += result.info.get("distance_travelled", 0.0)
        metrics.congestion_budget_used = smdp.budget.used
        metrics.protagonist_return = -(initial_demand - env.remaining_demand)  # placeholder; latency tracked in logs

    print("\nOpening PyGame. SPACE=pause, RIGHT=step, UP/DOWN=speed, ESC=quit.")
    renderer = PygameToyRenderer(fps=args.fps, sim_ticks_per_second=args.speed)
    try:
        running = True
        while running and not env.is_done() and env.time < cfg.max_ticks:
            if renderer.should_advance():
                one_tick()
            running = renderer.render(
                env, metrics,
                antagonist_action=last_cong,
                feed=[("info", f"t={env.time} delivered={metrics.total_delivery:.0f}/{initial_demand:.0f} "
                               f"budget={smdp.budget.used:.0f}/{cfg.congestion_budget:.0f}")],
            )
        metrics.done_reason = "done" if env.is_done() else "max_ticks"
        # Hold the final frame until the window is closed.
        while running:
            running = renderer.render(env, metrics, antagonist_action=last_cong)
    finally:
        renderer.close()
        print(f"\nClosed. Final: delivered {metrics.total_delivery:.0f}/{initial_demand:.0f}, "
              f"ticks {metrics.ticks}, budget {smdp.budget.used:.0f}.")


if __name__ == "__main__":
    main()
