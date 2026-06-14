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

    def solve(self, demand_dict: Dict[NodeId, float], num_trucks: int = 3, num_depots: int = 2) -> Tuple[List[NodeId], float]:
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
        self.model.eval()

        depot_pairs = list(itertools.combinations(self.node_list, num_depots))
        allocations = []
        for pair in depot_pairs:
            for alloc in itertools.combinations_with_replacement(pair, num_trucks):
                if len(set(alloc)) == num_depots:
                    allocations.append(alloc)

        if not allocations:
            raise RuntimeError("FLP solver was unable to locate a valid optimal configuration.")

        demands = [float(demand_dict.get(node, 0.0)) for node in self.node_list]
        
        all_features = []
        for alloc in allocations:
            truck_counts = [float(alloc.count(node)) for node in self.node_list]
            all_features.append(truck_counts + demands)
            
        features_tensor = torch.tensor(all_features, dtype=torch.float32)

        with torch.no_grad():
            predicted_costs = self.model(features_tensor).squeeze(-1)

        best_idx = int(torch.argmin(predicted_costs).item())
        best_cost = float(predicted_costs[best_idx].item())
        best_allocation = allocations[best_idx]

        return list(best_allocation), best_cost
