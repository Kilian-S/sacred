"""PyTorch Geometric and GATv2 Actor-Critic Network architectures for SACRED.

This module defines:
1. Featurization helpers to convert NetworkX/SMDP states into GNN-ready Tensors.
2. GATv2Encoder: Shared spatial representation learning backbone.
3. ProtagonistPolicyValueNet: Routing policy & critic for the dispatcher.
4. AntagonistPolicyValueNet: Edge congestion policy & critic for the adversary.
"""

from __future__ import annotations

from typing import Any, Mapping
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import numpy as np

_FEATURIZE_CACHE = {}

# Node feature width. Bumped 9 -> 11 for Stage 1.5 (request age + active truck's congestion-aware
# ETA; zero for static problems), and 11 -> 13 for the fixed hybrid rung: column 11 marks the
# active truck's assigned target (the goal it is routing toward — previously invisible to the
# policy) and column 12 is the congestion-aware distance-to-that-goal field (global routing
# information a 2-layer GNN cannot propagate itself; parity with greedy's Dijkstra). Both are zero
# whenever the observation lacks the keys, so pre-hybrid problems are informationally unchanged.
# Checkpoints trained at a narrower width are still evaluable: the SAC agents slice features to
# their own node_in_dim (new columns are appended LAST, so a [:, :11] slice reproduces the old
# featurization exactly).
NODE_FEATURE_DIM = 13
_WAIT_NORM = 100.0  # rough scale for request age (ticks) -> O(1)
_ETA_NORM = 50.0    # rough scale for congestion-aware ETA (graph diameter ~44) -> O(1)
_GOAL_NORM = 50.0   # same scale for the distance-to-goal field
# Full blockage sets effective_weight ~ distance/1e-6, so a node whose only route to the goal is
# blocked would get a ~1e4 feature after normalization — clamp distance features to a sane range.
_DIST_FEATURE_CAP = 10.0

def featurize_state(
    observation: dict[str, Any],
    active_truck_id: int | None = None,
) -> Data:
    """Convert an SMDP wrapper observation dict into a PyG Data object.

    Parameters
    ----------
    observation:
        The environment observation dict returned by SMDPWrapper or GraphEnv.
    active_truck_id:
        The ID of the truck that is currently making a decision. If None,
        truck-specific features will be set to zero.

    Returns
    -------
    Data:
        A PyTorch Geometric Data object with features:
        - x: Node features [num_nodes, 11]
        - edge_index: Graph topology [2, 2 * num_edges]
        - edge_attr: Edge features [2 * num_edges, 2]
    """
    nodes_dict = observation["nodes"]
    edges_dict = observation["edges"]
    trucks_dict = observation["trucks"]
    # Dynamic-queue features (Stage 1.5); empty/zero for static problems.
    node_waits = observation.get("node_waits", {})
    active_etas = observation.get("truck_etas", {}).get(active_truck_id, {}) if active_truck_id is not None else {}
    # Hybrid features: the active truck's assigned goal and the distance-to-goal field.
    goal_dists = observation.get("goal_dists", {}).get(active_truck_id, {}) if active_truck_id is not None else {}
    active_target = None
    if active_truck_id is not None and active_truck_id in trucks_dict:
        active_target = trucks_dict[active_truck_id].get("assigned_target")

    node_ids = sorted(list(nodes_dict.keys()))
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # 1. Normalize/Center coordinates
    cache_key = tuple(node_ids)
    if cache_key not in _FEATURIZE_CACHE:
        xs = [nodes_dict[n]["x"] for n in node_ids]
        ys = [nodes_dict[n]["y"] for n in node_ids]
        mean_x_val, std_x_val = np.mean(xs), np.std(xs) + 1e-6
        mean_y_val, std_y_val = np.mean(ys), np.std(ys) + 1e-6
        dists = [edata["distance"] for edata in edges_dict.values()]
        mean_dist_val = np.mean(dists) if dists else 1.0
        std_dist_val = np.std(dists) + 1e-6 if dists else 1.0

        # Precompute static edge index and normalized distances
        edge_indices = []
        norm_dists = []
        for (u, v), edata in sorted(list(edges_dict.items())):
            if u not in node_to_idx or v not in node_to_idx:
                continue
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            norm_dist = (edata["distance"] - mean_dist_val) / std_dist_val
            
            edge_indices.append([idx_u, idx_v])
            norm_dists.append(norm_dist)
            
            edge_indices.append([idx_v, idx_u])
            norm_dists.append(norm_dist)
            
        if edge_indices:
            edge_index_tensor_val = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            edge_index_tensor_val = torch.empty((2, 0), dtype=torch.long)

        _FEATURIZE_CACHE[cache_key] = (mean_x_val, std_x_val, mean_y_val, std_y_val, mean_dist_val, std_dist_val, edge_index_tensor_val, norm_dists)
    
    mean_x, std_x, mean_y, std_y, mean_dist, std_dist, edge_index_tensor, norm_dists = _FEATURIZE_CACHE[cache_key]

    # 2. Count truck presence and active target commitments
    trucks_per_node = {n: 0 for n in node_ids}
    targeted_by_other = {n: 0.0 for n in node_ids}
    other_targeted_capacity = {n: 0.0 for n in node_ids}

    for t_id, t in trucks_dict.items():
        # Count physical presence
        if t["current_node"] is not None:
            trucks_per_node[t["current_node"]] += 1

        # Count commitments by other trucks. A truck's commitment is its assigned request
        # (hybrid mode) falling back to its in-flight destination (destination mode — identical
        # to the old behaviour there, since assigned_target is always None outside hybrid; in
        # hybrid the destination is only the next hop and would be meaningless). When there is
        # no active truck (the ANTAGONIST's view), every truck counts as "other" — previously
        # the antagonist saw no commitments at all.
        if t_id != active_truck_id:
            commit = t.get("assigned_target") or t.get("destination")
            if commit is not None and commit in nodes_dict:
                targeted_by_other[commit] = 1.0
                other_targeted_capacity[commit] += t.get("capacity", 1.0)

    # 3. Build node features
    # Columns: [x_norm, y_norm, demand, is_depot, num_trucks, is_active_truck, active_truck_load,
    #           is_targeted_by_other, unassigned_demand, oldest_wait_norm, active_truck_eta_norm,
    #           is_active_target, goal_dist_norm]
    x_features = []
    for node_id in node_ids:
        ndata = nodes_dict[node_id]

        # Check if active truck is here
        is_active_here = 0.0
        active_load = 0.0
        if active_truck_id is not None and active_truck_id in trucks_dict:
            t = trucks_dict[active_truck_id]
            if t["current_node"] == node_id:
                is_active_here = 1.0
                active_load = t["load"] / max(1e-6, t["capacity"])

        node_demand = float(ndata["demand"])
        is_targeted = targeted_by_other[node_id]
        unassigned = max(0.0, node_demand - other_targeted_capacity[node_id])

        feat = [
            (ndata["x"] - mean_x) / std_x,
            (ndata["y"] - mean_y) / std_y,
            node_demand,
            1.0 if ndata["has_depot"] else 0.0,
            float(trucks_per_node[node_id]),
            is_active_here,
            active_load,
            is_targeted,
            unassigned,
            float(node_waits.get(node_id, 0.0)) / _WAIT_NORM,
            min(float(active_etas.get(node_id, 0.0)) / _ETA_NORM, _DIST_FEATURE_CAP),
            1.0 if (active_target is not None and node_id == active_target) else 0.0,
            min(float(goal_dists.get(node_id, 0.0)) / _GOAL_NORM, _DIST_FEATURE_CAP),
        ]
        x_features.append(feat)

    x_tensor = torch.tensor(x_features, dtype=torch.float32)

    # 4. Build dynamic edge features only (topology is cached)
    edge_features = []
    norm_idx = 0
    for (u, v), edata in sorted(list(edges_dict.items())):
        if u not in node_to_idx or v not in node_to_idx:
            continue
        
        cong = edata["congestion_level"]
        
        # Forward edge (u -> v)
        edge_features.append([norm_dists[norm_idx], cong])
        # Reverse edge (v -> u)
        edge_features.append([norm_dists[norm_idx+1], cong])
        
        norm_idx += 2

    if edge_features:
        edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_attr_tensor = torch.empty((0, 2), dtype=torch.float32)

    from torch_geometric.data import Batch
    data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    return Batch.from_data_list([data])


class GATv2Encoder(nn.Module):
    """Shared Graph Attention Backbone to learn node/edge spatial embeddings."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        from torch_geometric.nn import GATv2Conv

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Layer 1
        self.convs.append(
            GATv2Conv(
                in_channels=node_in_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=True,
                edge_dim=edge_in_dim,
                dropout=dropout,
            )
        )
        
        # Intermediate layers
        for _ in range(num_layers - 1):
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim * heads,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=True,
                    edge_dim=edge_in_dim,
                    dropout=dropout,
                )
            )

        # Output projection to standard hidden dim
        self.proj = nn.Linear(hidden_dim * heads, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.proj(h)


class ProtagonistPolicyValueNet(nn.Module):
    """Dispatcher Agent: maps GATv2 embeddings to node-routing logits and state value."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = GATv2Encoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        )
        
        # Bilinear attention scoring head for routing action selection
        self.policy_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

        # Critic state-value estimation layers
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_val = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        pyg_data: Data,
        active_node_idx: int,
        action_mask_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Perform forward pass to get routing action probabilities and state value.

        Parameters
        ----------
        pyg_data:
            Featurized PyG Data object.
        active_node_idx:
            Index of the active truck's current node in pyg_data.x.
        action_mask_indices:
            List of valid node indices that the active truck is allowed to travel to.

        Returns
        -------
        probs:
            Action probability tensor [len(action_mask_indices)] summing to 1.
        value:
            Estimated state value V(s) [1] for reinforcement learning.
        """
        # 1. Generate node embeddings: [num_nodes, hidden_dim]
        h = self.encoder(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        return self.head(h, active_node_idx, action_mask_indices)

    def head(
        self,
        h: torch.Tensor,
        active_node_idx: int,
        action_mask_indices: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy/value head on precomputed node embeddings ``h`` [num_nodes, hidden_dim].

        Separated from the encoder so :meth:`SAC.update` can batch-encode the whole
        minibatch in a single GATv2 pass and then apply this head per-sample. Behaviour
        is identical to running :meth:`forward` on the corresponding single graph.
        """
        # 2. Get active node embedding: [1, hidden_dim]
        h_active = h[active_node_idx].unsqueeze(0)

        # 3. If no actions are available, return dummy uniform prob and value
        if not action_mask_indices:
            probs = torch.empty(0, device=h.device)
            h_graph = torch.mean(h, dim=0, keepdim=True)
            v = self.critic_val(F.relu(self.critic_fc1(h_graph))).squeeze(-1)
            return probs, v

        # 4. Score valid destination nodes
        h_candidates = h[action_mask_indices]  # [num_candidates, hidden_dim]
        h_active_rep = h_active.expand(h_candidates.size(0), -1)  # [num_candidates, hidden_dim]

        logits = self.policy_bilinear(h_active_rep, h_candidates).squeeze(-1)  # [num_candidates]
        probs = F.softmax(logits, dim=-1)

        # 5. Critic value: Pool node embeddings to get graph state embedding
        h_graph = torch.mean(h, dim=0, keepdim=True)  # Global Average Pooling [1, hidden_dim]
        v = self.critic_val(F.relu(self.critic_fc1(h_graph))).squeeze(-1)  # [1]

        return probs, v


class AntagonistPolicyValueNet(nn.Module):
    """Adversary Agent: selects edges to congest and sets congestion levels."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        num_congestion_levels: int = 4,  # e.g., 0.25, 0.50, 0.75, 1.00
    ) -> None:
        super().__init__()
        self.encoder = GATv2Encoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        )

        # Edge feature processing MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Edge selection scoring head
        self.edge_scorer = nn.Linear(hidden_dim, 1)

        # Wait action scoring node
        self.wait_scorer = nn.Linear(hidden_dim, 1)

        # Congestion level selection head (conditioned on edge features)
        self.level_head = nn.Linear(hidden_dim, num_congestion_levels)

        # Critic layers
        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_val = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        pyg_data: Data,
        original_edges: list[tuple[Any, Any]],
        node_to_idx: Mapping[Any, int],
        allowed_edges: list[tuple[Any, Any]],
        remaining_budget: float,
        level_costs: list[float],  # maps index to cost
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass to get edge choice, wait choice, and level probability.

        Parameters
        ----------
        pyg_data:
            Featurized PyG Data.
        original_edges:
            List of undirected edges in the original env order.
        node_to_idx:
            Dict mapping NodeId to integer index in pyg_data.
        allowed_edges:
            Subset of original_edges that are NOT currently congested and can be chosen.
        remaining_budget:
            The antagonist's remaining congestion credit.
        level_costs:
            List of credit costs corresponding to each congestion level.

        Returns
        -------
        edge_probs:
            Probability tensor over allowed edges + [wait action] (size: len(allowed_edges) + 1).
        level_probs:
            Matrix of shape [len(allowed_edges), num_levels] containing valid level choices.
        value:
            Critic estimated state value V(s) [1].
        """
        h_nodes = self.encoder(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        return self.head(
            h_nodes, pyg_data.edge_index, pyg_data.edge_attr,
            node_to_idx, allowed_edges, remaining_budget, level_costs,
        )

    def head(
        self,
        h_nodes: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        node_to_idx: Mapping[Any, int],
        allowed_edges: list[tuple[Any, Any]],
        remaining_budget: float,
        level_costs: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Policy/value head on precomputed node embeddings (see ProtagonistPolicyValueNet.head)."""
        device = h_nodes.device
        num_allowed = len(allowed_edges)

        h_graph = torch.mean(h_nodes, dim=0, keepdim=True)  # [1, hidden_dim]

        # 2. Critic state value estimation
        v = self.critic_val(F.relu(self.critic_fc1(h_graph))).squeeze(-1)

        # 3. Calculate edge embeddings for all allowed edges
        # We find the node indices of edge endpoints, gather their embeddings, and append current edge features.
        h_edges = []
        # Re-construct maps of existing edge features for lookup
        # edge_index is directed (2 * num_edges), we lookup the undirected pairs
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        edge_features_dict = dict(zip(zip(u_list, v_list), edge_attr))

        for u, v_node in allowed_edges:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v_node]
            
            emb_u = h_nodes[idx_u]
            emb_v = h_nodes[idx_v]
            # Lookup original edge attribute tensor
            attr = edge_features_dict.get((idx_u, idx_v), torch.zeros(2, device=device))
            
            # Enforce permutation invariance for undirected edges
            if idx_u < idx_v:
                emb_min, emb_max = emb_u, emb_v
            else:
                emb_min, emb_max = emb_v, emb_u

            # Combine [emb_min, emb_max, edge_attr]
            combined = torch.cat([emb_min, emb_max, attr], dim=-1)
            h_edges.append(combined)

        # Handle the case where no edge changes are allowed (e.g. out of budget)
        if num_allowed == 0:
            edge_probs = torch.ones(1, device=device)  # 100% probability on "wait"
            level_probs = torch.empty((0, len(level_costs)), device=device)
            return edge_probs, level_probs, v

        # Tensorize edge encodings: [num_allowed, hidden_dim * 2 + 2]
        h_edges_tensor = torch.stack(h_edges, dim=0)
        h_edge_features = self.edge_mlp(h_edges_tensor)  # [num_allowed, hidden_dim]

        # 4. Compute edge logits and level logits
        edge_logits = self.edge_scorer(h_edge_features).squeeze(-1)  # [num_allowed]
        wait_logit = self.wait_scorer(h_graph).squeeze(-1)  # [1]

        # Combine logits: [allowed_edges... , wait]
        all_logits = torch.cat([edge_logits, wait_logit], dim=0)  # [num_allowed + 1]

        # 5. Compute congestion level probabilities for each allowed edge
        level_logits = self.level_head(h_edge_features)  # [num_allowed, num_levels]
        
        # Apply action masking for levels based on the remaining budget
        # If cost of a level exceeds budget, mask to -infinity
        level_mask = torch.tensor(
            [cost <= remaining_budget + 1e-6 for cost in level_costs],
            dtype=torch.bool,
            device=device
        )
        
        # If no levels are affordable, mask out all edge actions to force wait
        if not level_mask.any():
            all_logits[:-1] = -1e9

        edge_probs = F.softmax(all_logits, dim=-1)

        # Expand mask to [num_allowed, num_levels]
        level_mask_expanded = level_mask.unsqueeze(0).expand(num_allowed, -1)
        masked_level_logits = level_logits.masked_fill(~level_mask_expanded, -1e9)
        level_probs = F.softmax(masked_level_logits, dim=-1)

        return edge_probs, level_probs, v
