import pickle
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

pkl_path = '/Users/kilian/Kilian/ICL/Thesis/code/data_gen/shared_data/kaliningrad/kaliningrad_network.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

g = data['graph']
node_coords = data['node_coords']
edge_attrs = data['edge_attrs']

# 1. Convert to NetworkX MultiDiGraph
G_nx = nx.MultiDiGraph()
for i, c in enumerate(node_coords):
    # c is (lat, lon), osmnx expects x=lon, y=lat
    G_nx.add_node(i, y=c[0], x=c[1])

for i, edge in enumerate(g.es):
    s, t = edge.tuple
    attrs = edge_attrs[i] if edge_attrs else {}
    G_nx.add_edge(s, t, **attrs)
    # If the original igraph is undirected, we might need to add the reverse edge
    if not g.is_directed():
        G_nx.add_edge(t, s, **attrs)

print(f"Original nx graph: {len(G_nx.nodes)} nodes, {len(G_nx.edges)} edges")

G_nx.graph["crs"] = "epsg:4326"

# 2. Project and Consolidate
G_proj = ox.project_graph(G_nx)
G_cons = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=30, dead_ends=False)
G_cons = ox.project_graph(G_cons, to_crs='epsg:4326')

print(f"Consolidated nx graph: {len(G_cons.nodes)} nodes, {len(G_cons.edges)} edges")

# 3. Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

def plot_nx(graph, ax, title, color='blue'):
    xs = [data['x'] for node, data in graph.nodes(data=True)]
    ys = [data['y'] for node, data in graph.nodes(data=True)]
    
    ax.scatter(xs, ys, s=15, c='red', zorder=2)
    
    for u, v, data in graph.edges(data=True):
        x_coords = [graph.nodes[u]['x'], graph.nodes[v]['x']]
        y_coords = [graph.nodes[u]['y'], graph.nodes[v]['y']]
        ax.plot(x_coords, y_coords, c=color, linewidth=2, zorder=1, alpha=0.6)
        
    ax.set_title(f"{title}\nNodes: {len(graph.nodes)}, Edges: {len(graph.edges)}", fontsize=16)
    ax.set_aspect('equal')
    ax.axis('off')

plot_nx(G_nx, ax1, "Original Kaliningrad Graph", color='blue')
plot_nx(G_cons, ax2, "Consolidated Graph (Tolerance=30m)", color='purple')

plt.tight_layout()
out_path = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/kaliningrad_consolidated_compare.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved comparison to {out_path}")
