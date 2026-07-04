import pickle
import networkx as nx
import osmnx as ox

pkl_path = '/Users/kilian/Kilian/ICL/Thesis/code/data_gen/shared_data/kaliningrad/kaliningrad_network.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

g = data['graph']
node_coords = data['node_coords']
edge_attrs = data['edge_attrs']

G_nx = nx.MultiDiGraph()
for i, c in enumerate(node_coords):
    G_nx.add_node(i, y=c[0], x=c[1])

for i, edge in enumerate(g.es):
    s, t = edge.tuple
    attrs = dict(edge_attrs[i]) if edge_attrs else {}
    if 'geometry' in attrs:
        del attrs['geometry']
    G_nx.add_edge(s, t, **attrs)
    if not g.is_directed():
        G_nx.add_edge(t, s, **attrs)

G_nx.graph["crs"] = "epsg:4326"

# Project and Consolidate at 30m
G_proj = ox.project_graph(G_nx)
G_cons = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=30, dead_ends=False)
G_cons = ox.project_graph(G_cons, to_crs='epsg:4326')

for u, v, k, data in G_cons.edges(data=True, keys=True):
    if 'geometry' in data:
        del data['geometry']

# Export to GeoJSON
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_cons)

# Clean up data types for GeoJSON serialization (sometimes lists/dicts cause errors)
for col in edges_gdf.columns:
    if col != 'geometry':
        edges_gdf[col] = edges_gdf[col].astype(str)

out_nodes = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/data/maps/kaliningrad_nodes.geojson'
out_edges = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/data/maps/kaliningrad_edges.geojson'

nodes_gdf.to_file(out_nodes, driver='GeoJSON')
edges_gdf.to_file(out_edges, driver='GeoJSON')

print(f"Successfully exported {len(nodes_gdf)} nodes to {out_nodes}")
print(f"Successfully exported {len(edges_gdf)} edges to {out_edges}")
