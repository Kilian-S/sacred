import matplotlib.pyplot as plt
import networkx as nx
from src.utils.graph_utils import load_osm_graph_and_demands

nodes_path = 'data/maps/kaliningrad_simplified_30m/kaliningrad_nodes.geojson'
edges_path = 'data/maps/kaliningrad_simplified_30m/kaliningrad_edges.geojson'
tasks_path = 'data/maps/koenigsberg1.json'

nodes, edges = load_osm_graph_and_demands(nodes_path, edges_path, tasks_path)

# Build a quick NetworkX graph just for plotting
G = nx.Graph()
for node_id, data in nodes.items():
    G.add_node(node_id, x=data['lon'], y=data['lat'], demand=data['base_demand'])

for u, v, attrs in edges:
    G.add_edge(u, v)

fig, ax = plt.subplots(figsize=(15, 15))

# Plot edges
xs = []
ys = []
for u, v in G.edges():
    xs.extend([G.nodes[u]['x'], G.nodes[v]['x'], None])
    ys.extend([G.nodes[u]['y'], G.nodes[v]['y'], None])
ax.plot(xs, ys, color='lightgray', linewidth=1.5, zorder=1)

# Plot nodes based on demand
node_xs = []
node_ys = []
node_sizes = []
node_colors = []

for node_id, data in G.nodes(data=True):
    demand = data['demand']
    node_xs.append(data['x'])
    node_ys.append(data['y'])
    
    if demand == 0:
        node_sizes.append(10)
        node_colors.append('black')
    else:
        # Scale size by demand (e.g. 1 package = size 50, 10 packages = size 500)
        node_sizes.append(50 + (demand * 20))
        node_colors.append('red')

# Plot zeros first, then demands on top
zero_mask = [s == 10 for s in node_sizes]
demand_mask = [s > 10 for s in node_sizes]

# Plot empty intersections
ax.scatter(
    [node_xs[i] for i in range(len(node_xs)) if zero_mask[i]],
    [node_ys[i] for i in range(len(node_ys)) if zero_mask[i]],
    s=15, c='black', alpha=0.3, zorder=2, label='No Demand'
)

# Plot demand hotspots
scatter = ax.scatter(
    [node_xs[i] for i in range(len(node_xs)) if demand_mask[i]],
    [node_ys[i] for i in range(len(node_ys)) if demand_mask[i]],
    s=[node_sizes[i] for i in range(len(node_sizes)) if demand_mask[i]],
    c='red', alpha=0.6, zorder=3, edgecolors='darkred', label='Demand (Size = Volume)'
)

ax.set_title("Kaliningrad Base Demand Heatmap\n(1000 Packages Snapped to 290 Intersections)", fontsize=18)
ax.set_aspect('equal')
ax.axis('off')
ax.legend(loc='lower right', fontsize=14)

plt.tight_layout()
out_path = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/kaliningrad_demand_heatmap.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved heatmap to {out_path}")
