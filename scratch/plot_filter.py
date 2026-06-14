import pickle
import matplotlib.pyplot as plt

pkl_path = '/Users/kilian/Kilian/ICL/Thesis/code/data_gen/shared_data/kaliningrad/kaliningrad_network.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

g = data['graph']
node_coords = data['node_coords']
edge_attrs = data['edge_attrs']

ALLOWED_HIGHWAYS = {'primary', 'secondary', 'tertiary', 'trunk', 'motorway', 'primary_link', 'secondary_link', 'tertiary_link', 'trunk_link', 'motorway_link'}

edges_to_keep = []
for i, attrs in enumerate(edge_attrs):
    highway = attrs.get('highway', '')
    if isinstance(highway, list):
        if any(h in ALLOWED_HIGHWAYS for h in highway):
            edges_to_keep.append(i)
    else:
        if highway in ALLOWED_HIGHWAYS:
            edges_to_keep.append(i)

# Create filtered graph
g_filtered = g.subgraph_edges(edges_to_keep, delete_vertices=False)

# Remove isolated nodes to get the true node count
nodes_with_edges = [v.index for v in g_filtered.vs if v.degree() > 0]
g_filtered = g_filtered.subgraph(nodes_with_edges)
# Also need to filter coords
filtered_coords = [node_coords[i] for i in nodes_with_edges]

# Plotting
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

def plot_igraph(graph, coords, ax, title, color='blue'):
    xs = [c[1] for c in coords]
    ys = [c[0] for c in coords]
    
    ax.scatter(xs, ys, s=15, c='red', zorder=2)
    
    for edge in graph.es:
        s, t = edge.tuple
        ax.plot([xs[s], xs[t]], [ys[s], ys[t]], c=color, linewidth=2, zorder=1, alpha=0.6)
        
    ax.set_title(f"{title}\nNodes: {graph.vcount()}, Edges: {graph.ecount()}", fontsize=16)
    ax.set_aspect('equal')
    ax.axis('off')

plot_igraph(g, node_coords, ax1, "Original Kaliningrad Graph", color='blue')
plot_igraph(g_filtered, filtered_coords, ax2, "Filtered 'Arteries' Graph", color='green')

plt.tight_layout()
out_path = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/kaliningrad_filter_compare.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved comparison to {out_path}")
