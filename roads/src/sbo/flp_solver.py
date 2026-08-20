"""Facility Location Problem solver logic using Surrogate-Based Optimization (SBO)."""

from __future__ import annotations

import itertools
from typing import Any, Dict, Hashable, List, Tuple

import torch
from src.env.graph_env import NodeId


class FLPSolver:
    """Solver for the Facility Location Problem leveraging a trained Surrogate model."""

    def __init__(self, surrogate_model: torch.nn.Module, node_list: List[NodeId]) -> None:
        """Initialize the FLP Solver.

        Parameters
        ----------
        surrogate_model:
            Trained PyTorch SurrogateMLP model.
        node_list:
            Ordered list of node IDs in the graph network, matching the feature representation order.
        """
        self.model = surrogate_model
        self.node_list = node_list

    def solve(self, demand_dict: Dict[NodeId, float], num_trucks: int = 3, num_depots: int = 2, K: int = None) -> Tuple[List[NodeId], float]:
        """Find the optimal fleet allocation that minimizes predicted expected adversarial cost.

        Parameters
        ----------
        demand_dict:
            Mapping of node ID to its demand value.
        num_trucks:
            Total number of trucks in the fleet.
        num_depots:
            Number of distinct depots to place.

        Returns
        -------
        Tuple of (list_of_truck_starting_nodes, predicted_cost).
        """
        if K is not None:
            num_depots = K
            num_trucks = K

        self.model.eval()

        depot_pairs = list(itertools.combinations(self.node_list, num_depots))
        allocations = []
        remaining_trucks = num_trucks - num_depots
        if remaining_trucks >= 0:
            for pair in depot_pairs:
                base_alloc = list(pair)
                for extra in itertools.combinations_with_replacement(pair, remaining_trucks):
                    allocations.append(tuple(base_alloc + list(extra)))

        if not allocations:
            raise RuntimeError("FLP solver was unable to locate a valid optimal configuration.")

        num_allocs = len(allocations)
        num_nodes = len(self.node_list)
        node_to_idx = {node: idx for idx, node in enumerate(self.node_list)}
        demands = [float(demand_dict.get(node, 0.0)) for node in self.node_list]
        
        device = next(self.model.parameters()).device
        features_tensor = torch.zeros((num_allocs, num_nodes * 2), dtype=torch.float32, device=device)
        demands_tensor = torch.tensor(demands, dtype=torch.float32, device=device)
        
        for i, alloc in enumerate(allocations):
            for node in alloc:
                features_tensor[i, node_to_idx[node]] += 1.0
            features_tensor[i, num_nodes:] = demands_tensor

        with torch.no_grad():
            predicted_costs = self.model(features_tensor).squeeze(-1)

        best_idx = int(torch.argmin(predicted_costs).item())
        best_cost = float(predicted_costs[best_idx].item())
        best_allocation = allocations[best_idx]

        return list(best_allocation), best_cost
