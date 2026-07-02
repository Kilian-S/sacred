import argparse
import sys
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.env.multi_agent import GameConfig, SacredToyGame
from src.envs.osm_factory import make_osm_env
from src.env.renderer import PygameToyRenderer
from src.agents.sac import ProtagonistSAC, AntagonistSAC
from src.env.smdp_wrapper import SMDPDecisionWrapper, SMDPConfig

class DummyPolicy:
    def __init__(self):
        self.action = {}
    def act(self, env, *args, **kwargs):
        return self.action

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="Run without UI")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print("1. Initializing Neural Networks...")
    config = SMDPConfig(
        max_ticks=600,
        antagonist_interval=30,
        congestion_duration=30,
        congestion_budget=500.0,
        congestion_cooldown=0,
        remaining_demand_penalty=0.5,
        delivery_reward=10.0,
        time_penalty=1.0,
        congestion_cost=0.1,
        congestion_levels=(0.25, 0.5, 0.75, 1.0)
    )
    
    protag = ProtagonistSAC(
        node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4,
        lr_actor=5e-5, lr_critic=1e-3, gamma=0.99, tau=0.005, alpha_init=1.0,
        autotune_alpha=True, target_entropy=-1.0, device="cpu"
    )
    protag.actor.load_state_dict(torch.load(PROJECT_ROOT / 'models/protagonist/actor.pt', weights_only=True))
    protag.actor.eval()

    antag = AntagonistSAC(
        node_in_dim=11, edge_in_dim=2, hidden_dim=64, num_layers=2, heads=4,
        num_congestion_levels=len(config.congestion_levels),
        level_costs=[level * config.congestion_duration for level in config.congestion_levels],
        lr_actor=5e-5, lr_critic=1e-3, gamma=0.99, tau=0.005, alpha_init=1.0,
        autotune_alpha=True, target_entropy=-1.0, device="cpu"
    )
    antag.actor.load_state_dict(torch.load(PROJECT_ROOT / 'models/antagonist/actor.pt', weights_only=True))
    antag.actor.eval()
    
    print("2. Initializing the Kaliningrad OSM Environment...")
    game_config = GameConfig(
        max_ticks=600,
        congestion_budget=500.0,
        congestion_level=1.0,
        congestion_duration=30,
        congestion_cooldown=0,
    )
    
    game = SacredToyGame(
        env_factory=lambda: make_osm_env(num_trucks=4, truck_capacity=40.0, episode_packages=150),
        config=game_config,
    )
    game.reset()
    
    dummy_protag = DummyPolicy()
    dummy_antag = DummyPolicy()
    game.protagonist = dummy_protag
    game.antagonist = dummy_antag
    
    smdp = SMDPDecisionWrapper(
        env_factory=lambda: game.env,
        config=config,
    )
    
    print(f"-> Environment ready! Nodes: {len(game.env.graph.nodes)}, Edges: {len(game.env.graph.edges)}")
    print("3. Launching 4 Trucks with Co-Evolutionary AI...")

    if not args.headless:
        renderer = PygameToyRenderer(width=1400, height=900, fps=60, sim_ticks_per_second=20.0)
    
    running = True
    latest = None
    
    smdp.env = game.env
    smdp.budget = game.budget
    smdp.active_congestion = game.active_congestion
    smdp.cooldown_remaining = game.cooldown_remaining
    
    while running and not game.env.is_done() and game.metrics.ticks < game_config.max_ticks:
        advance = args.headless or renderer.should_advance()
        if advance:
            obs = game.env.observe()
            
            # Sync smdp state to game state
            smdp.env = game.env
            smdp.budget = game.budget
            smdp.active_congestion = game.active_congestion
            smdp.cooldown_remaining = game.cooldown_remaining
            
            # --- Protagonist Action ---
            p_mask = smdp.protagonist_action_mask()
            waiting_trucks = [tid for tid, opts in p_mask.items() if opts]
            p_actions = {}
            if waiting_trucks:
                projected_obs = dict(obs)
                projected_obs["trucks"] = {tid: dict(t) for tid, t in obs["trucks"].items()}
                for truck_id in waiting_trucks:
                    projected_obs["active_truck"] = truck_id
                    projected_obs["allowed_destinations"] = {"protagonist": dict(p_mask)}
                    
                    truck_action = protag.select_action(projected_obs, p_mask, deterministic=True)
                    p_actions.update(truck_action)
                    
                    chosen_node = truck_action.get(truck_id)
                    if chosen_node is not None:
                        projected_obs["trucks"][truck_id]["destination"] = chosen_node
                        projected_obs["trucks"][truck_id]["current_node"] = None
                        
            # --- Antagonist Action ---
            a_action = {}
            if game.env.time >= smdp.next_antagonist_tick:
                a_mask = smdp.antagonist_action_mask()
                remaining_budget = smdp.budget.remaining
                a_action_tuple = antag.select_action(obs, a_mask, remaining_budget, deterministic=False)
                print(f"[DEBUG] t={game.env.time} Antagonist woke up. Mask size: {len(a_mask)}. Selected: {a_action_tuple}")
                if a_action_tuple:
                    a_action = {a_action_tuple[0]: a_action_tuple[1]}
                    smdp.next_antagonist_tick += smdp.config.antagonist_interval
            
            dummy_protag.action = p_actions
            dummy_antag.action = a_action
            
            latest = game.step()
            
        if not args.headless:
            running = renderer.render(
                game.env,
                latest.metrics if latest is not None else game.metrics,
                protagonist_reward=latest.protagonist_reward if latest is not None else 0.0,
                antagonist_reward=latest.antagonist_reward if latest is not None else 0.0,
                antagonist_action=latest.antagonist_action if latest is not None else {}
            )
            
    if args.headless:
        print(f"Finished {game.metrics.ticks} ticks! Delivered: {game.metrics.total_delivery}")

if __name__ == '__main__':
    main()
