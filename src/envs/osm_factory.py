import os
from pathlib import Path
from src.env.graph_env import GraphEnv
from src.utils.graph_utils import load_osm_graph_and_demands, generate_stochastic_demand

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def make_osm_env(
    nodes_path: str = str(PROJECT_ROOT / 'data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson'),
    edges_path: str = str(PROJECT_ROOT / 'data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson'),
    tasks_path: str = str(PROJECT_ROOT / 'data/maps/koenigsberg1.json'),
    depots: list[str] = ['284', '39'],
    num_trucks: int = 4,
    truck_capacity: float = 40.0,
    episode_packages: int = 150
) -> GraphEnv:
    """
    Factory function to build a GraphEnv using the Kaliningrad OSM data.
    """
    # 1. Load the raw OSM graph and base heatmap
    nodes_dict, edges_list = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    
    # 2. Sample a random Micro-Shift (e.g., 150 packages) based on the heatmap
    nodes_dict = generate_stochastic_demand(nodes_dict, total_episode_packages=episode_packages)
    
    # 3. Inject the static Depots
    for depot_id in depots:
        if depot_id in nodes_dict:
            nodes_dict[depot_id]['has_depot'] = True
            
    # Distribute trucks evenly across the available depots
    starting_nodes = [depots[i % len(depots)] for i in range(num_trucks)]

    # 4. Initialize the PyTorch Environment
    env = GraphEnv(
        nodes=nodes_dict,
        edges=edges_list,
        num_trucks=num_trucks,
        truck_capacity=truck_capacity,
        truck_starting_nodes=starting_nodes,
        truck_speed=1.0,  # 1 edge unit distance per tick
        max_time=600
    )
        
    return env
