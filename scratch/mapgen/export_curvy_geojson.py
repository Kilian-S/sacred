import pickle
import networkx as nx
import osmnx as ox
from shapely.geometry import LineString

pkl_path = '/Users/kilian/Kilian/ICL/Thesis/code/data_gen/shared_data/kaliningrad/kaliningrad_network.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

g = data['graph']
node_coords = data['node_coords']
edge_attrs = data['edge_attrs']
edge_geom = data.get('edge_geom')

G_nx = nx.MultiDiGraph()
for i, c in enumerate(node_coords):
    # c is (lat, lon) -> x=lon, y=lat
    G_nx.add_node(i, y=c[0], x=c[1])

for i, edge in enumerate(g.es):
    s, t = edge.tuple
    attrs = dict(edge_attrs[i]) if edge_attrs else {}
    
    # Add the curvy geometry!
    if edge_geom and i < len(edge_geom) and edge_geom[i]:
        # edge_geom[i] is a list of (lon, lat) tuples
        coords = edge_geom[i]
        if len(coords) >= 2:
            attrs['geometry'] = LineString(coords)
            
    G_nx.add_edge(s, t, **attrs)
    if not g.is_directed():
        # If undirected, the reverse edge geometry should probably be reversed
        attrs_rev = dict(attrs)
        if 'geometry' in attrs_rev:
            attrs_rev['geometry'] = LineString(list(reversed(coords)))
        G_nx.add_edge(t, s, **attrs_rev)

G_nx.graph["crs"] = "epsg:4326"

nodes_gdf, edges_gdf = ox.graph_to_gdfs(G_nx)

# Clean up data types
for col in edges_gdf.columns:
    if col != 'geometry':
        edges_gdf[col] = edges_gdf[col].astype(str)

out_nodes = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/data/maps/kaliningrad_original_curvy/kaliningrad_nodes.geojson'
out_edges = '/Users/kilian/Kilian/ICL/Thesis/code/sacred/data/maps/kaliningrad_original_curvy/kaliningrad_edges.geojson'

nodes_gdf.to_file(out_nodes, driver='GeoJSON')
edges_gdf.to_file(out_edges, driver='GeoJSON')

print(f"Successfully exported {len(nodes_gdf)} curvy nodes to {out_nodes}")
print(f"Successfully exported {len(edges_gdf)} curvy edges to {out_edges}")
