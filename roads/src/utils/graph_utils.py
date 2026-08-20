import json
import numpy as np
from scipy.spatial import KDTree

def load_osm_graph_and_demands(nodes_path: str, edges_path: str, tasks_path: str):
    """Parse osmnx GeoJSON exports into the nodes dict and edge list GraphEnv expects."""
    with open(nodes_path, 'r') as f:
        nodes_geojson = json.load(f)
    
    with open(edges_path, 'r') as f:
        edges_geojson = json.load(f)
        
    with open(tasks_path, 'r') as f:
        tasks = json.load(f)

    nodes = {}
    node_coords = []
    node_id_list = []
    
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
        
    edges = []
    for feature in edges_geojson['features']:
        props = feature['properties']
        u = str(props['u'])
        v = str(props['v'])
        
        # Travel weight is in ticks at 100 m per tick; a missing length defaults to 100 m.
        val = props.get('length')
        length_m = float(val) if val is not None else 100.0
        weight = length_m / 100.0
        
        # Floor at one tick so no edge can be traversed instantaneously.
        weight = max(1.0, round(weight, 1))
        
        edges.append((u, v, {'distance': weight}))

    tree = KDTree(node_coords)
    
    for task in tasks:
        lat = task['lat']
        lon = task['lon']
        dist, idx = tree.query([lat, lon])
        nearest_node_id = node_id_list[idx]
        
        # 1 task = 1 package
        nodes[nearest_node_id]['base_demand'] += 1.0

    return nodes, edges

def generate_stochastic_demand(nodes: dict, total_episode_packages: int = 150):
    """Sample a per-episode demand realisation totalling ``total_episode_packages`` packages.

    Base demands are normalised into a probability distribution and packages drawn from it
    multinomially, so the episode total is fixed while its spatial pattern varies.
    """
    node_ids = list(nodes.keys())
    base_demands = np.array([nodes[n]['base_demand'] for n in node_ids])
    
    total_base = np.sum(base_demands)
    if total_base > 0:
        probs = base_demands / total_base
    else:
        probs = np.ones(len(base_demands)) / len(base_demands)
        
    sampled_packages = np.random.multinomial(total_episode_packages, probs)

    for idx, node_id in enumerate(node_ids):
        nodes[node_id]['demand'] = float(sampled_packages[idx])
        
    return nodes
