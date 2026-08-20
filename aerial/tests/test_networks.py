"""Unit tests for the GATv2 network architectures and state featurisation."""

import unittest
import torch

from src.env.toy_graph import make_toy_graph_env
from src.agents.device import get_torch_device
from src.agents.networks import (
    featurize_state,
    ProtagonistPolicyValueNet,
    AntagonistPolicyValueNet,
)


class TestNetworks(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_toy_graph_env(num_trucks=2)
        self.obs = self.env.observe()

    def test_featurize_state(self) -> None:
        pyg_data = featurize_state(self.obs, active_truck_id=0)

        # 9 nodes in the toy graph, each with 14 features
        self.assertEqual(pyg_data.x.shape, (9, 14))

        # 15 undirected edges = 30 directed edges in PyG
        self.assertEqual(pyg_data.edge_index.shape[0], 2)
        self.assertEqual(pyg_data.edge_index.shape[1], 30)

        # Edge features: norm_distance, congestion_level, and vulnerability in column 4
        self.assertEqual(pyg_data.edge_attr.shape, (30, 5))

        # Active truck (id 0) is at "depot" initially
        node_ids = sorted(list(self.obs["nodes"].keys()))
        depot_idx = node_ids.index("depot")
        
        # Check active truck flag (column 5) and load (column 6)
        self.assertEqual(pyg_data.x[depot_idx, 5].item(), 1.0)
        self.assertEqual(pyg_data.x[depot_idx, 6].item(), 1.0)

        for idx in range(9):
            if idx != depot_idx:
                self.assertEqual(pyg_data.x[idx, 5].item(), 0.0)
                self.assertEqual(pyg_data.x[idx, 6].item(), 0.0)

    def test_protagonist_network_forward(self) -> None:
        device = get_torch_device()
        pyg_data = featurize_state(self.obs, active_truck_id=0).to(device)

        net = ProtagonistPolicyValueNet(
            node_in_dim=14,
            edge_in_dim=5,
            hidden_dim=32,
            num_layers=2,
            heads=2,
        ).to(device)

        node_ids = sorted(list(self.obs["nodes"].keys()))
        depot_idx = node_ids.index("depot")

        # the depot's neighbours in the toy graph
        allowed_destinations = ["a", "d", "hub"]
        allowed_indices = [node_ids.index(n) for n in allowed_destinations]

        probs, val = net(pyg_data, active_node_idx=depot_idx, action_mask_indices=allowed_indices)

        self.assertEqual(probs.shape, (3,))
        self.assertEqual(val.shape, (1,))

        self.assertAlmostEqual(torch.sum(probs).item(), 1.0, places=5)
        self.assertTrue(torch.all(probs >= 0.0))

    def test_antagonist_network_forward(self) -> None:
        device = get_torch_device()
        pyg_data = featurize_state(self.obs, active_truck_id=None).to(device)

        num_levels = 4
        level_costs = [level * 12.0 for level in [0.25, 0.50, 0.75, 1.00]]

        net = AntagonistPolicyValueNet(
            node_in_dim=14,
            edge_in_dim=5,
            hidden_dim=32,
            num_layers=2,
            heads=2,
            num_congestion_levels=num_levels,
        ).to(device)

        node_ids = sorted(list(self.obs["nodes"].keys()))
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        original_edges = list(self.obs["edges"].keys())

        allowed_edges = original_edges[:5]
        
        # a high budget leaves every level affordable
        edge_probs, level_probs, val = net(
            pyg_data=pyg_data,
            original_edges=original_edges,
            node_to_idx=node_to_idx,
            allowed_edges=allowed_edges,
            remaining_budget=100.0,
            level_costs=level_costs,
        )

        # one entry per allowed edge, plus the wait action
        self.assertEqual(edge_probs.shape, (6,))
        self.assertEqual(level_probs.shape, (5, num_levels))
        self.assertEqual(val.shape, (1,))

        self.assertAlmostEqual(torch.sum(edge_probs).item(), 1.0, places=5)
        self.assertTrue(torch.all(edge_probs >= 0.0))

        for i in range(5):
            self.assertAlmostEqual(torch.sum(level_probs[i]).item(), 1.0, places=5)
            self.assertTrue(torch.all(level_probs[i] >= 0.0))

        # budget masking: the levels cost [3.0, 6.0, 9.0, 12.0], so a budget of 4.0 leaves only
        # the first affordable and the rest must carry zero probability
        _, level_probs_low, _ = net(
            pyg_data=pyg_data,
            original_edges=original_edges,
            node_to_idx=node_to_idx,
            allowed_edges=allowed_edges,
            remaining_budget=4.0,
            level_costs=level_costs,
        )

        for i in range(5):
            self.assertAlmostEqual(level_probs_low[i, 0].item(), 1.0, places=5)
            self.assertAlmostEqual(torch.sum(level_probs_low[i, 1:]).item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
