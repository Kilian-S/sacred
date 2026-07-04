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
    # Strip geometry to ensure straight lines in EPSG:4326 and avoid projection errors
    if 'geometry' in attrs:
        del attrs['geometry']
    G_nx.add_edge(s, t, **attrs)
    if not g.is_directed():
        G_nx.add_edge(t, s, **attrs)

G_nx.graph["crs"] = "epsg:4326"

# Export to GeoJSON directly without consolidation
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_nx)

# Clean up data types for GeoJSON serialization
for col in edges_gdf.columns:
    if col != 'geometry':
        edges_gdf[col] = edges_gdf[col].astype(str)

out_nodes = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/data/maps/kaliningrad_original/kaliningrad_nodes.geojson'
out_edges = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/data/maps/kaliningrad_original/kaliningrad_edges.geojson'

nodes_gdf.to_file(out_nodes, driver='GeoJSON')
edges_gdf.to_file(out_edges, driver='GeoJSON')

print(f"Successfully exported original {len(nodes_gdf)} nodes to {out_nodes}")
print(f"Successfully exported original {len(edges_gdf)} edges to {out_edges}")
