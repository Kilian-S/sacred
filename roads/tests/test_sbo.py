"""Unit tests for SurrogateMLP and FLPSolver implementations."""

import unittest
import torch
import numpy as np

from src.sbo.surrogate import SurrogateMLP, train_surrogate
from src.sbo.flp_solver import FLPSolver


class TestSBO(unittest.TestCase):
    def test_surrogate_mlp_forward(self) -> None:
        # 9 depot indicators + 9 demand values = 18 input features
        input_dim = 18
        model = SurrogateMLP(input_dim=input_dim, hidden_dim=16)

        # Batch of 4 samples
        dummy_x = torch.randn(4, input_dim)
        output = model(dummy_x)

        self.assertEqual(output.shape, (4, 1))

    def test_train_surrogate(self) -> None:
        np.random.seed(42)
        torch.manual_seed(42)

        input_dim = 10
        num_samples = 16

        features = np.random.randn(num_samples, input_dim).astype(np.float32)
        targets = np.random.randn(num_samples).astype(np.float32)

        model, losses = train_surrogate(
            features,
            targets,
            epochs=5,
            lr=0.01,
            batch_size=4,
            hidden_dim=8,
        )

        self.assertIsInstance(model, SurrogateMLP)
        self.assertEqual(len(losses), 5)
        self.assertTrue(all(loss >= 0 for loss in losses))

    def test_flp_solver(self) -> None:
        torch.manual_seed(42)
        node_list = ["a", "b", "c", "d"]
        num_nodes = len(node_list)

        # 4 depot indicators + 4 demands = 8 dimensions
        model = SurrogateMLP(input_dim=num_nodes * 2, hidden_dim=8)

        solver = FLPSolver(surrogate_model=model, node_list=node_list)
        demands = {"a": 1.0, "b": 0.5, "c": 2.0}

        optimal_depot_1, predicted_cost_1 = solver.solve(demands, K=1)
        self.assertEqual(len(optimal_depot_1), 1)
        self.assertIn(optimal_depot_1[0], node_list)
        self.assertIsInstance(predicted_cost_1, float)

        optimal_depots_2, predicted_cost_2 = solver.solve(demands, K=2)
        self.assertEqual(len(optimal_depots_2), 2)
        for depot in optimal_depots_2:
            self.assertIn(depot, node_list)
        self.assertIsInstance(predicted_cost_2, float)
