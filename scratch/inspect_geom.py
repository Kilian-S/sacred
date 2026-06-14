import pickle

pkl_path = '/Users/kilian/Kilian/ICL/Thesis/code/data_gen/shared_data/kaliningrad/kaliningrad_network.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

edge_attrs = data['edge_attrs']
edge_geom = data.get('edge_geom')

print("edge_geom type:", type(edge_geom))
if edge_geom:
    print("first edge_geom:", edge_geom[0])

print("first edge_attrs geometry:", edge_attrs[0].get('geometry'))
