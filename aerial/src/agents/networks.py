"""GATv2 actor-critic network architectures for SACRED.

Holds the featurisation helpers that turn SMDP observations into PyTorch Geometric tensors, the
shared GATv2 encoder backbone, and the two policy-value heads: a routing policy and critic for the
dispatcher, and an edge-congestion policy and critic for the adversary.
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


def node_index_map(observation: dict[str, Any]) -> dict[Any, int]:
    """Node id to row index in the tensors ``featurize_state`` builds for this observation.

    Rows are ordered by ``sorted(nodes.keys())``, so every consumer that indexes into the
    featurised node matrix must build its map with this helper. Dict insertion order differs from
    sorted order on the OSM graphs, and relying on it silently reads the wrong rows.
    """
    return {nid: idx for idx, nid in enumerate(sorted(observation["nodes"].keys()))}

# Node feature width. Columns absent from an observation are filled with zero. New columns are
# always appended LAST, and the SAC agents slice features down to their own node_in_dim, so a
# checkpoint trained at a narrower width stays evaluable.
NODE_FEATURE_DIM = 14
_WAIT_NORM = 100.0  # rough scale for request age (ticks) -> O(1)
_ETA_NORM = 50.0    # rough scale for congestion-aware ETA (graph diameter ~44) -> O(1)
_GOAL_NORM = 50.0   # same scale for the distance-to-goal field
# Full blockage sets effective_weight ~ distance/1e-6, so a node whose only route to the goal is
# blocked would reach ~1e4 after normalisation. Clamp distance features to a sane range.
_DIST_FEATURE_CAP = 10.0

# Edge feature width. Columns 2/3 are the DIRECTED truck occupancy (count, furthest progress
# fraction), the motion state an adversary needs to attack ahead of a moving truck; column 4 is the
# edge's interception vulnerability p_e from the instance's threat map, zero when absent. Same
# append-last and slicing rule as the node features.
EDGE_FEATURE_DIM = 5

def featurize_state(
    observation: dict[str, Any],
    active_truck_id: int | None = None,
) -> Data:
    """Convert an SMDP wrapper observation dict into a PyG Data object.

    Args:
        observation: The observation dict returned by SMDPWrapper or GraphEnv.
        active_truck_id: The truck currently making a decision. None zeroes the truck-specific
            features.

    Returns:
        A batched PyG Data object with node features x [num_nodes, NODE_FEATURE_DIM], topology
        edge_index [2, 2 * num_edges] and edge features edge_attr [2 * num_edges,
        EDGE_FEATURE_DIM].
    """
    nodes_dict = observation["nodes"]
    edges_dict = observation["edges"]
    trucks_dict = observation["trucks"]
    # Dynamic-queue features; empty or zero for static problems.
    node_waits = observation.get("node_waits", {})
    active_etas = observation.get("truck_etas", {}).get(active_truck_id, {}) if active_truck_id is not None else {}
    goal_dists = observation.get("goal_dists", {}).get(active_truck_id, {}) if active_truck_id is not None else {}
    # Per-node fraction of earlier convoys routed through it.
    taken_node_frac = observation.get("taken_node_frac", {})
    active_target = None
    if active_truck_id is not None and active_truck_id in trucks_dict:
        active_target = trucks_dict[active_truck_id].get("assigned_target")

    node_ids = sorted(list(nodes_dict.keys()))
    node_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}
    num_nodes = len(node_ids)

    # The cache key covers the nodes AND the edge set's size and total length: two graphs sharing
    # node ids but differing in edges or lengths must not share cached edge indices or
    # normalisation constants.
    cache_key = (tuple(node_ids), len(edges_dict),
                 round(sum(float(e["distance"]) for e in edges_dict.values()), 6))
    if cache_key not in _FEATURIZE_CACHE:
        xs = [nodes_dict[n]["x"] for n in node_ids]
        ys = [nodes_dict[n]["y"] for n in node_ids]
        mean_x_val, std_x_val = np.mean(xs), np.std(xs) + 1e-6
        mean_y_val, std_y_val = np.mean(ys), np.std(ys) + 1e-6
        dists = [edata["distance"] for edata in edges_dict.values()]
        mean_dist_val = np.mean(dists) if dists else 1.0
        std_dist_val = np.std(dists) + 1e-6 if dists else 1.0

        # Precompute the static edge index and normalised distances.
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

    trucks_per_node = {n: 0 for n in node_ids}
    targeted_by_other = {n: 0.0 for n in node_ids}
    other_targeted_capacity = {n: 0.0 for n in node_ids}

    for t_id, t in trucks_dict.items():
        if t["current_node"] is not None:
            trucks_per_node[t["current_node"]] += 1

        # A truck's commitment is its assigned request, falling back to its in-flight destination.
        # assigned_target is always None outside hybrid mode; inside it the destination is only the
        # next hop and would be meaningless. With no active truck (the antagonist's view) every
        # truck counts as "other".
        if t_id != active_truck_id:
            commit = t.get("assigned_target") or t.get("destination")
            if commit is not None and commit in nodes_dict:
                targeted_by_other[commit] = 1.0
                other_targeted_capacity[commit] += t.get("capacity", 1.0)

    # Columns: [x_norm, y_norm, demand, is_depot, num_trucks, is_active_truck, active_truck_load,
    #           is_targeted_by_other, unassigned_demand, oldest_wait_norm, active_truck_eta_norm,
    #           is_active_target, goal_dist_norm, taken_node_frac]
    x_features = []
    for node_id in node_ids:
        ndata = nodes_dict[node_id]

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
            float(taken_node_frac.get(node_id, 0.0)),
        ]
        x_features.append(feat)

    x_tensor = torch.tensor(x_features, dtype=torch.float32)

    # Truck occupancy per DIRECTED edge, keyed by travel direction (truck.edge = (from, to)), so a
    # truck's heading is encoded by which directed row carries the value.
    edge_occ: dict[tuple[int, int], list[float]] = {}
    for t in trucks_dict.values():
        e = t.get("edge")
        if not e:
            continue
        eu, ev = e
        if eu not in node_to_idx or ev not in node_to_idx:
            continue
        edata = edges_dict.get((eu, ev)) or edges_dict.get((ev, eu))
        dist = float(edata["distance"]) if edata else 1.0
        frac = min(1.0, max(0.0, float(t.get("edge_progress", 0.0)) / max(dist, 1e-9)))
        rec = edge_occ.setdefault((node_to_idx[eu], node_to_idx[ev]), [0.0, 0.0])
        rec[0] += 1.0                # occupancy count on this directed edge
        rec[1] = max(rec[1], frac)   # furthest along (0 = just entered, ~1 = about to arrive)

    # Dynamic edge features only; topology is cached. Columns: [norm_distance, congestion,
    # occupancy_count(directed), max_progress_frac(directed), vulnerability]
    edge_vuln = observation.get("edge_vulnerability", {})
    edge_features = []
    norm_idx = 0
    _no_occ = (0.0, 0.0)
    for (u, v), edata in sorted(list(edges_dict.items())):
        if u not in node_to_idx or v not in node_to_idx:
            continue

        cong = edata["congestion_level"]
        fwd = edge_occ.get((node_to_idx[u], node_to_idx[v]), _no_occ)
        rev = edge_occ.get((node_to_idx[v], node_to_idx[u]), _no_occ)
        vuln = float(edge_vuln.get((u, v), edge_vuln.get((v, u), 0.0)))

        edge_features.append([norm_dists[norm_idx], cong, fwd[0], fwd[1], vuln])      # u -> v
        edge_features.append([norm_dists[norm_idx + 1], cong, rev[0], rev[1], vuln])  # v -> u

        norm_idx += 2

    if edge_features:
        edge_attr_tensor = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_attr_tensor = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    from torch_geometric.data import Batch
    data = Data(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    return Batch.from_data_list([data])


def _route_head_terms(net: nn.Module, logits: torch.Tensor, action_mask_indices) -> torch.Tensor:
    """Apply menu-head discriminability terms undiluted at the head.

    Two optional terms, applied only when the trainer has attached the corresponding attributes in
    menu-select mode; absent attributes leave behaviour byte-identical. ``route_feats`` [R, F] are
    static per-route features carrying a learned weight ``route_feat_w`` [F], shifting each logit
    by feats @ w. ``route_bias`` [R] is a learned per-route scalar bias giving pure identity
    capacity. Both stay Bellman-consistent when the same attributes are attached to the Q heads.
    """
    feats = getattr(net, "route_feats", None)
    fw = getattr(net, "route_feat_w", None)
    if feats is not None and fw is not None:
        idx = torch.as_tensor(list(action_mask_indices), dtype=torch.long, device=logits.device)
        logits = logits + feats[idx] @ fw
    bias = getattr(net, "route_bias", None)
    if bias is not None:
        idx = torch.as_tensor(list(action_mask_indices), dtype=torch.long, device=logits.device)
        logits = logits + bias[idx]
    return logits


class GATv2Encoder(nn.Module):
    """Shared graph-attention backbone learning node and edge spatial embeddings."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 4,
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

        self.proj = nn.Linear(hidden_dim * heads, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = x
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr=edge_attr)
            h = F.elu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        return self.proj(h)


class ProtagonistPolicyValueNet(nn.Module):
    """Dispatcher agent mapping GATv2 embeddings to node-routing logits and a state value."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 4,
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
        
        self.policy_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

        self.critic_fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.critic_val = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        pyg_data: Data,
        active_node_idx: int,
        action_mask_indices: list[int],
        taken: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode the graph and return routing action probabilities and the state value.

        Args:
            active_node_idx: Index of the active truck's current node in pyg_data.x.
            action_mask_indices: Node indices the active truck may travel to.

        Returns:
            probs [len(action_mask_indices)] summing to 1, and the state value V(s) [1].
        """
        h = self.encoder(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        return self.head(h, active_node_idx, action_mask_indices, taken)

    def head(
        self,
        h: torch.Tensor,
        active_node_idx: int,
        action_mask_indices: list[int],
        taken: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy and value head on precomputed node embeddings ``h`` [num_nodes, hidden_dim].

        Separated from the encoder so :meth:`SAC.update` can batch-encode a whole minibatch in one
        GATv2 pass and apply this head per sample, identically to :meth:`forward` on the
        corresponding single graph.
        """
        h_active = h[active_node_idx].unsqueeze(0)

        # No actions available: report an empty distribution alongside the value.
        if len(action_mask_indices) == 0:
            probs = torch.empty(0, device=h.device)
            h_graph = torch.mean(h, dim=0, keepdim=True)
            v = self.critic_val(F.relu(self.critic_fc1(h_graph))).squeeze(-1)
            return probs, v

        # Candidates are scored per node (destination or first hop) by default. In route
        # menu-select mode, set by attaching `menu_routes`, `action_mask_indices` are route ids
        # instead and each route is scored by the mean-pooled embedding of its nodes, which handles
        # overlapping routes and reuses the bilinear head without extra parameters.
        menu = getattr(self, "menu_routes", None)
        if menu is not None:
            h_candidates = torch.stack([h[menu[int(r)]].mean(dim=0) for r in action_mask_indices])
        else:
            h_candidates = h[action_mask_indices]  # [num_candidates, hidden_dim]
        h_active_rep = h_active.expand(h_candidates.size(0), -1)  # [num_candidates, hidden_dim]

        logits = self.policy_bilinear(h_active_rep, h_candidates).squeeze(-1)  # [num_candidates]
        fw = getattr(self, "follow_w", None)
        if fw is not None and taken is not None:
            # Route-correlation shift delivered straight to the head rather than through the GNN,
            # so the follower can generalise "follow the signal" to non-modal routes.
            logits = logits + fw * taken
        logits = _route_head_terms(self, logits, action_mask_indices)
        probs = F.softmax(logits, dim=-1)

        h_graph = torch.mean(h, dim=0, keepdim=True)  # global average pooling, [1, hidden_dim]
        v = self.critic_val(F.relu(self.critic_fc1(h_graph))).squeeze(-1)  # [1]

        return probs, v


class AntagonistPolicyValueNet(nn.Module):
    """Adversary agent selecting edges to congest and setting congestion levels."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 4,
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

        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.edge_scorer = nn.Linear(hidden_dim, 1)
        self.wait_scorer = nn.Linear(hidden_dim, 1)
        self.level_head = nn.Linear(hidden_dim, num_congestion_levels)

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
        """Encode the graph and return the edge choice, wait choice, and level probabilities.

        Args:
            original_edges: Undirected edges in the original environment order.
            node_to_idx: Node id to row index in pyg_data.
            allowed_edges: The subset of original_edges not currently congested.
            remaining_budget: The antagonist's remaining congestion credit.
            level_costs: Credit cost of each congestion level.

        Returns:
            edge_probs over allowed edges plus a trailing wait action, level_probs
            [len(allowed_edges), num_levels], and the state value V(s) [1].
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
        """Policy and value head on precomputed node embeddings.

        See :meth:`ProtagonistPolicyValueNet.head` for why the encoder is split off.
        """
        device = h_nodes.device
        num_allowed = len(allowed_edges)

        h_graph = torch.mean(h_nodes, dim=0, keepdim=True)  # [1, hidden_dim]
        v = self.critic_val(F.relu(self.critic_fc1(h_graph))).squeeze(-1)

        # Edge embeddings: gather each endpoint's embedding and append the current edge features.
        # edge_index is directed (2 * num_edges), so the undirected pairs are looked up in both
        # orientations.
        h_edges = []
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        edge_features_dict = dict(zip(zip(u_list, v_list), edge_attr))

        for u, v_node in allowed_edges:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v_node]
            
            emb_u = h_nodes[idx_u]
            emb_v = h_nodes[idx_v]
            attr = edge_features_dict.get((idx_u, idx_v), torch.zeros(edge_attr.size(1), device=device))

            # Order the endpoints by index so the encoding is permutation-invariant.
            if idx_u < idx_v:
                emb_min, emb_max = emb_u, emb_v
            else:
                emb_min, emb_max = emb_v, emb_u

            combined = torch.cat([emb_min, emb_max, attr], dim=-1)
            h_edges.append(combined)

        # No edge change is allowed, for instance when out of budget.
        if num_allowed == 0:
            edge_probs = torch.ones(1, device=device)  # 100% probability on "wait"
            level_probs = torch.empty((0, len(level_costs)), device=device)
            return edge_probs, level_probs, v

        h_edges_tensor = torch.stack(h_edges, dim=0)  # [num_allowed, hidden_dim * 2 + edge_in_dim]
        h_edge_features = self.edge_mlp(h_edges_tensor)  # [num_allowed, hidden_dim]

        edge_logits = self.edge_scorer(h_edge_features).squeeze(-1)  # [num_allowed]
        wait_logit = self.wait_scorer(h_graph).squeeze(-1)  # [1]
        all_logits = torch.cat([edge_logits, wait_logit], dim=0)  # [num_allowed + 1], wait last

        level_logits = self.level_head(h_edge_features)  # [num_allowed, num_levels]

        # Levels costing more than the remaining budget are masked out.
        level_mask = torch.tensor(
            [cost <= remaining_budget + 1e-6 for cost in level_costs],
            dtype=torch.bool,
            device=device
        )
        
        # With nothing affordable, every edge action is masked so only wait survives.
        if not level_mask.any():
            all_logits[:-1] = -1e9

        edge_probs = F.softmax(all_logits, dim=-1)

        level_mask_expanded = level_mask.unsqueeze(0).expand(num_allowed, -1)
        masked_level_logits = level_logits.masked_fill(~level_mask_expanded, -1e9)
        level_probs = F.softmax(masked_level_logits, dim=-1)

        return edge_probs, level_probs, v
