"""Unit tests for the GATv2 neural network architectures and state featurization."""

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
        # Create a toy graph environment with 2 trucks
        self.env = make_toy_graph_env(num_trucks=2)
        self.obs = self.env.observe()

    def test_featurize_state(self) -> None:
        pyg_data = featurize_state(self.obs, active_truck_id=0)

        # 9 nodes in toy graph, each has 14 features (col 14 = multi-convoy route-correlation)
        self.assertEqual(pyg_data.x.shape, (9, 14))

        # 15 undirected edges = 30 directed edges in PyG
        self.assertEqual(pyg_data.edge_index.shape[0], 2)
        self.assertEqual(pyg_data.edge_index.shape[1], 30)

        # Edge features (norm_distance, congestion_level)
        self.assertEqual(pyg_data.edge_attr.shape, (30, 4))

        # Active truck feature checks
        # Active truck (id 0) is at "depot" initially
        node_ids = sorted(list(self.obs["nodes"].keys()))
        depot_idx = node_ids.index("depot")
        
        # Check active truck flag (column 5) and load (column 6)
        self.assertEqual(pyg_data.x[depot_idx, 5].item(), 1.0)
        self.assertEqual(pyg_data.x[depot_idx, 6].item(), 1.0)

        # Other nodes should not have active truck flag
        for idx in range(9):
            if idx != depot_idx:
                self.assertEqual(pyg_data.x[idx, 5].item(), 0.0)
                self.assertEqual(pyg_data.x[idx, 6].item(), 0.0)

    def test_protagonist_network_forward(self) -> None:
        device = get_torch_device()
        pyg_data = featurize_state(self.obs, active_truck_id=0).to(device)

        net = ProtagonistPolicyValueNet(
            node_in_dim=14,
            edge_in_dim=4,
            hidden_dim=32,
            num_layers=2,
            heads=2,
        ).to(device)

        node_ids = sorted(list(self.obs["nodes"].keys()))
        depot_idx = node_ids.index("depot")

        # In toy graph, depot neighbors are "a", "d", "hub"
        allowed_destinations = ["a", "d", "hub"]
        allowed_indices = [node_ids.index(n) for n in allowed_destinations]

        probs, val = net(pyg_data, active_node_idx=depot_idx, action_mask_indices=allowed_indices)

        # Output validations
        self.assertEqual(probs.shape, (3,))
        self.assertEqual(val.shape, (1,))

        # Probabilities must sum to 1
        self.assertAlmostEqual(torch.sum(probs).item(), 1.0, places=5)
        self.assertTrue(torch.all(probs >= 0.0))

    def test_antagonist_network_forward(self) -> None:
        device = get_torch_device()
        pyg_data = featurize_state(self.obs, active_truck_id=None).to(device)

        # Define 4 discrete levels: 0.25, 0.50, 0.75, 1.00
        num_levels = 4
        level_costs = [level * 12.0 for level in [0.25, 0.50, 0.75, 1.00]]

        net = AntagonistPolicyValueNet(
            node_in_dim=14,
            edge_in_dim=4,
            hidden_dim=32,
            num_layers=2,
            heads=2,
            num_congestion_levels=num_levels,
        ).to(device)

        node_ids = sorted(list(self.obs["nodes"].keys()))
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
        original_edges = list(self.obs["edges"].keys())

        # Select a subset of edges that can be congested (e.g. first 5 edges)
        allowed_edges = original_edges[:5]
        
        # Test with high budget (all levels allowed)
        edge_probs, level_probs, val = net(
            pyg_data=pyg_data,
            original_edges=original_edges,
            node_to_idx=node_to_idx,
            allowed_edges=allowed_edges,
            remaining_budget=100.0,
            level_costs=level_costs,
        )

        # Output validations
        # edge_probs has shape [len(allowed_edges) + 1] (includes wait action)
        self.assertEqual(edge_probs.shape, (6,))
        # level_probs has shape [len(allowed_edges), num_levels]
        self.assertEqual(level_probs.shape, (5, num_levels))
        self.assertEqual(val.shape, (1,))

        self.assertAlmostEqual(torch.sum(edge_probs).item(), 1.0, places=5)
        self.assertTrue(torch.all(edge_probs >= 0.0))

        # Each edge's level probabilities should sum to 1
        for i in range(5):
            self.assertAlmostEqual(torch.sum(level_probs[i]).item(), 1.0, places=5)
            self.assertTrue(torch.all(level_probs[i] >= 0.0))

        # Test budget masking (e.g. remaining budget is very low, so only level 1 is allowed)
        # Level 1 cost = 0.25 * 12 = 3.0. Let's set budget = 4.0.
        # level_costs are: [3.0, 6.0, 9.0, 12.0]
        # Levels 2, 3, 4 cost > budget, so they should have 0 probability.
        _, level_probs_low, _ = net(
            pyg_data=pyg_data,
            original_edges=original_edges,
            node_to_idx=node_to_idx,
            allowed_edges=allowed_edges,
            remaining_budget=4.0,
            level_costs=level_costs,
        )

        # Check that levels 1, 2, 3 index (1, 2, 3) are zero/masked
        for i in range(5):
            self.assertAlmostEqual(level_probs_low[i, 0].item(), 1.0, places=5)
            self.assertAlmostEqual(torch.sum(level_probs_low[i, 1:]).item(), 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
