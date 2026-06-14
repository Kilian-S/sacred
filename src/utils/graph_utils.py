import json
import numpy as np
from scipy.spatial import KDTree

def load_osm_graph_and_demands(nodes_path: str, edges_path: str, tasks_path: str):
    """
    Universal parser for data_gen + osmnx GeoJSON exports.
    Creates the nodes dict and edges list for GraphEnv.
    """
    with open(nodes_path, 'r') as f:
        nodes_geojson = json.load(f)
    
    with open(edges_path, 'r') as f:
        edges_geojson = json.load(f)
        
    with open(tasks_path, 'r') as f:
        tasks = json.load(f)

    nodes = {}
    node_coords = []
    node_id_list = []
    
    # 1. Parse Nodes
    for feature in nodes_geojson['features']:
        node_id = str(feature['properties']['osmid'])
        coords = feature['geometry']['coordinates'] # [lon, lat]
        node_id_list.append(node_id)
        # Store as [lat, lon] for KDTree distance math
        node_coords.append([coords[1], coords[0]]) 
        
        nodes[node_id] = {
            'demand': 0.0,
            'base_demand': 0.0,
            'has_depot': False,
            'y': coords[1],
            'x': coords[0],
            'lat': coords[1],
            'lon': coords[0]
        }
        
    # 2. Parse Edges
    edges = []
    for feature in edges_geojson['features']:
        props = feature['properties']
        u = str(props['u'])
        v = str(props['v'])
        
        # We need the edge length to calculate travel ticks
        val = props.get('length')
        length_m = float(val) if val is not None else 100.0
        weight = length_m / 100.0
        
        # Ensure minimum weight of 1 tick to prevent instant teleportation
        weight = max(1.0, round(weight, 1))
        
        edges.append((u, v, {'distance': weight}))

    # 3. Snap Demands using KDTree
    tree = KDTree(node_coords)
    
    for task in tasks:
        lat = task['lat']
        lon = task['lon']
        # Find nearest node
        dist, idx = tree.query([lat, lon])
        nearest_node_id = node_id_list[idx]
        
        # 1 task = 1 package
        nodes[nearest_node_id]['base_demand'] += 1.0

    return nodes, edges

def generate_stochastic_demand(nodes: dict, total_episode_packages: int = 150):
    """
    Takes the Base Demands (which sum to ~1000) and stochastically samples a Micro-Shift.
    Scales the total to total_episode_packages using a probability distribution.
    """
    node_ids = list(nodes.keys())
    base_demands = np.array([nodes[n]['base_demand'] for n in node_ids])
    
    # Calculate probability distribution
    total_base = np.sum(base_demands)
    if total_base > 0:
        probs = base_demands / total_base
    else:
        probs = np.ones(len(base_demands)) / len(base_demands)
        
    # Sample packages according to the probabilities
    sampled_packages = np.random.multinomial(total_episode_packages, probs)
    
    # Apply to nodes dict
    for idx, node_id in enumerate(node_ids):
        nodes[node_id]['demand'] = float(sampled_packages[idx])
        
    return nodes
