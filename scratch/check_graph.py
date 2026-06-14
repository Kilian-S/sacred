import pickle

pkl_path = '/Users/kilian/Kilian/ICL/Thesis/code/data_gen/shared_data/kaliningrad/kaliningrad_network.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f)

print("Type of data:", type(data))
if isinstance(data, dict):
    print("Keys in data:", data.keys())
    if 'node_coords' in data:
        print("node_coords length:", len(data['node_coords']))
        print("first 3 coords:", data['node_coords'][:3])
