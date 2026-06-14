from src.utils.graph_utils import load_osm_graph_and_demands, generate_stochastic_demand

def test_pipeline():
    nodes_path = 'data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson'
    edges_path = 'data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson'
    tasks_path = 'data/maps/koenigsberg1.json'
    
    print("1. Loading Data Pipeline...")
    nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)
    
    print(f"-> Successfully parsed {len(nodes)} nodes and {len(edges)} edges.")
    
    # Check total base demand
    total_base_demand = sum([data['base_demand'] for data in nodes.values()])
    print(f"-> Snapped exactly {total_base_demand} packages to the graph intersections.")
    
    print("\n2. Testing Micro-Shift Generator...")
    nodes = generate_stochastic_demand(nodes, total_episode_packages=150)
    
    total_spawned = sum([data['demand'] for data in nodes.values()])
    print(f"-> Stochastic shift generated exactly {total_spawned} packages for this episode.")
    
    # Print a heavy demand node as proof
    hotspot = max(nodes.items(), key=lambda x: x[1]['demand'])
    print(f"-> Episode Hotspot: Node {hotspot[0]} has {hotspot[1]['demand']} packages.")

if __name__ == '__main__':
    test_pipeline()
