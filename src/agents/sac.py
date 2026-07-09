"""Soft Actor-Critic (SAC) implementation for SACRED.

This module provides two distinct agent managers optimized for the discrete,
masked Graph Attention Networks:
1. ProtagonistSAC: Manages fleet routing decisions with A* path selection.
2. AntagonistSAC: Manages dynamic edge congestion and duration selection under credit budgets.

Both classes support Semi-Markov Decision Process (SMDP) temporal discounting (gamma^dt).
"""

from __future__ import annotations

import math
import random
from collections import deque
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.data import Batch, Data

from src.agents.device import get_torch_device
from src.agents.networks import (
    AntagonistPolicyValueNet,
    ProtagonistPolicyValueNet,
    featurize_state,
    node_index_map,
)


def _collate_graphs(data_list: list) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    """Concatenate per-sample graphs into one disjoint graph for a single encoder pass.

    Each entry is a single-graph PyG object (as returned by ``featurize_state``). Node
    features are concatenated, edge indices are offset per graph, and ``offsets`` marks
    each graph's node span so the encoder output can be sliced back per sample. Because
    the union has no cross-graph edges, GATv2 message passing is identical to encoding
    each graph separately (guarded by tests/test_batched_equivalence.py).
    """
    xs, eis, eas = [], [], []
    offsets = [0]
    for d in data_list:
        n = d.x.size(0)
        xs.append(d.x)
        eis.append(d.edge_index + offsets[-1])
        eas.append(d.edge_attr)
        offsets.append(offsets[-1] + n)
    x = torch.cat(xs, dim=0)
    edge_index = torch.cat(eis, dim=1) if eis else torch.empty((2, 0), dtype=torch.long)
    edge_attr = torch.cat(eas, dim=0)
    return x, edge_index, edge_attr, offsets


def _clip_x(x: torch.Tensor, node_in_dim: int) -> torch.Tensor:
    """Slice node features down to the width an agent's networks were built for (no-op when they
    already match). New feature columns are appended last, so this exactly reproduces the older
    featurization for checkpoints trained before a width bump."""
    return x[:, :node_in_dim] if x.size(1) > node_in_dim else x


def _clip_ea(edge_attr: torch.Tensor, edge_in_dim: int) -> torch.Tensor:
    """Edge-feature counterpart of :func:`_clip_x` (edge columns are also appended last)."""
    return edge_attr[:, :edge_in_dim] if edge_attr.size(1) > edge_in_dim else edge_attr


def infer_node_in_dim(actor_state_dict: Mapping[str, Any], default: int = 13) -> int:
    """Read the node-feature width a checkpoint was trained with from its first GATv2 layer.

    featurize_state appends new columns LAST, so an agent built with the inferred (narrower)
    node_in_dim slices current features down to exactly what the checkpoint saw in training —
    this keeps pre-hybrid checkpoints (11-dim, e.g. gen02_dynassign) evaluable after the 13-dim
    bump without a separate legacy featurizer.
    """
    for key in ("encoder.convs.0.lin_l.weight", "encoder.convs.0.lin_r.weight"):
        w = actor_state_dict.get(key)
        if w is not None:
            return int(w.shape[1])
    return default


def infer_edge_in_dim(actor_state_dict: Mapping[str, Any], default: int = 4) -> int:
    """Edge-feature counterpart of :func:`infer_node_in_dim` (GATv2's lin_edge input width)."""
    w = actor_state_dict.get("encoder.convs.0.lin_edge.weight")
    return int(w.shape[1]) if w is not None else default


def _cached_featurize(trans: Any, key: str, build_fn):
    """Memoize a featurized graph on the transition, building it once via ``build_fn``.

    Tolerates transitions deserialized from older pickles (e.g. preseed
    ``data/erb_transitions.pt``) whose ``feature_cache`` slot was never assigned: the
    slot exists on the class, so we lazily initialize it here.
    """
    fc = getattr(trans, "feature_cache", None)
    if fc is None:
        fc = {}
        trans.feature_cache = fc
    val = fc.get(key)
    if val is None:
        val = build_fn()
        fc[key] = val
    return val


class ReplayBuffer:
    """Experience replay buffer for off-policy transition storage."""

    def __init__(self, capacity: int = 50000) -> None:
        self.buffer: deque[Any] = deque(maxlen=capacity)

    def push(self, transition: Any) -> None:
        """Add a transition tuple to the buffer."""
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> list[Any]:
        """Randomly sample a batch of transitions."""
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class ProtagonistQNet(nn.Module):
    """Twin Q-network implementation for the Protagonist (Dispatcher)."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
    ) -> None:
        super().__init__()
        from src.agents.networks import GATv2Encoder

        self.encoder = GATv2Encoder(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        )
        self.q_bilinear = nn.Bilinear(hidden_dim, hidden_dim, 1)

    def forward(
        self,
        pyg_data: Data,
        active_node_idx: int,
        action_mask_indices: list[int],
    ) -> torch.Tensor:
        """Calculate state-action Q-values for all candidate destinations."""
        h = self.encoder(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        return self.head(h, active_node_idx, action_mask_indices)

    def head(
        self,
        h: torch.Tensor,
        active_node_idx: int,
        action_mask_indices: list[int],
        taken: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Q-head on precomputed node embeddings ``h`` (see ProtagonistPolicyValueNet.head)."""
        if len(action_mask_indices) == 0:
            return torch.empty(0, device=h.device)

        h_active = h[active_node_idx].unsqueeze(0)
        menu = getattr(self, "menu_routes", None)
        if menu is not None:  # ROUTE menu-select: mean-pool each route's node embeddings
            h_candidates = torch.stack([h[menu[int(r)]].mean(dim=0) for r in action_mask_indices])
        else:
            h_candidates = h[action_mask_indices]
        h_active_rep = h_active.expand(h_candidates.size(0), -1)

        q_values = self.q_bilinear(h_active_rep, h_candidates).squeeze(-1)  # [num_candidates]
        fw = getattr(self, "follow_w", None)
        if fw is not None and taken is not None:
            # LEVER 2 (critic half): give Q a direct, un-GNN-diluted dependence on 'did the leader
            # take this route' as a LEARNED input (Bellman-consistent), so it can rank the leader's
            # route and the actor gets a gradient to grow follow_w. The missing half of the fix.
            q_values = q_values + fw * taken
        return q_values


class ProtagonistSAC:
    """Soft Actor-Critic agent for the fleet routing Protagonist."""

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 1.0,
        autotune_alpha: bool = True,
        buffer_capacity: int = 50000,
        device: str | None = None,
        reward_scale: float = 0.001,
        target_entropy: float | None = None,
        role_alpha: bool = False,
        lr_alpha: float | None = None,
        alpha_floor: float | None = None,
        legacy_next_alpha: bool = False,
    ) -> None:
        self.device = device or get_torch_device()
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.target_entropy = target_entropy
        # Optional SECOND temperature for a per-decision "role" (multi-convoy: convoy-0 leader vs
        # followers). A single alpha cannot hold the leader near max-entropy while driving followers
        # to ~0, so a transition tagged state["alpha_group"]==1 uses this follower alpha in BOTH the
        # actor loss and its own auto-tuning. Default off -> single-alpha behaviour is byte-identical.
        self.role_alpha = role_alpha
        # Optional FLOOR on the primary temperature: after each auto-tune step, clamp log_alpha so
        # alpha cannot collapse below alpha_floor (toward a deterministic, exploitable policy). In
        # multi-convoy fleet-route mode the primary alpha IS the leader's, so this is the leader-alpha
        # floor that kills the across-seed variance from the leader over-concentrating in bad seeds
        # (mirrors the follower-alpha late-decay lesson). Default None = no floor = byte-identical.
        self.alpha_floor = alpha_floor
        # gen10-MC2 isolation flag: True reverts the 2026-07-09 role-alpha TARGET fix (V(s')
        # entropy term always uses the primary alpha, the pre-fix behaviour) while keeping the
        # node-ordering fix; used to attribute the gen10-MC regression. Default False = fixed.
        self.legacy_next_alpha = legacy_next_alpha
        # Feature widths this agent's networks consume. featurize_state may emit MORE columns
        # (new ones are appended last); _clip_x/_clip_ea slice down so checkpoints trained at a
        # narrower width keep seeing byte-identical inputs (see infer_node_in_dim/infer_edge_in_dim).
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim

        # 1. Initialize Actor & twin Critics
        self.actor = ProtagonistPolicyValueNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        ).to(self.device)

        self.q1 = ProtagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        ).to(self.device)

        self.q2 = ProtagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        ).to(self.device)

        # 2. Target Critics
        self.target_q1 = ProtagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())

        self.target_q2 = ProtagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
        ).to(self.device)
        self.target_q2.load_state_dict(self.q2.state_dict())

        # 3. Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr_critic,
        )

        # 4. Temperature Entropy coefficient (alpha)
        self.autotune_alpha = autotune_alpha
        alr = lr_alpha if lr_alpha is not None else lr_actor  # temperature LR (role_alpha wants it fast)
        if autotune_alpha:
            self.log_alpha = torch.tensor(
                math.log(alpha_init),
                requires_grad=True,
                device=self.device,
                dtype=torch.float32,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alr)
            self.alpha = self.log_alpha.exp().item()
            self.alpha_foll = self.alpha  # always present; overridden below when role_alpha is on
            if self.role_alpha:
                self.log_alpha_foll = torch.tensor(
                    math.log(alpha_init), requires_grad=True, device=self.device, dtype=torch.float32)
                self.alpha_foll_optimizer = optim.Adam([self.log_alpha_foll], lr=alr)
                self.alpha_foll = self.log_alpha_foll.exp().item()
        else:
            self.alpha = alpha_init
            self.alpha_foll = alpha_init

        # 5. Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(
        self,
        observation: dict[str, Any],
        action_mask: dict[int, list[Any]],
        deterministic: bool = False,
    ) -> dict[int, Any]:
        """Select a destination node for the active truck using the policy.

        Parameters
        ----------
        observation:
            SMDP observation dict.
        action_mask:
            Dict mapping active truck ID to a list of allowed destination nodes.
        deterministic:
            If True, choose the highest-probability action.

        Returns
        -------
        dict[int, Any]:
            A dispatch dictionary `{active_truck_id: chosen_node}`.
        """
        active_truck = observation.get("active_truck")
        if active_truck is None or active_truck not in action_mask:
            return {}

        allowed_nodes = action_mask[active_truck]
        if not allowed_nodes:
            return {}

        # 1. Featurize state and find node indices
        pyg_data = featurize_state(observation, active_truck).to(self.device)
        pyg_data.x = _clip_x(pyg_data.x, self.node_in_dim)
        pyg_data.edge_attr = _clip_ea(pyg_data.edge_attr, self.edge_in_dim)
        node_to_idx = node_index_map(observation)  # MUST match featurize_state's row order

        active_idx = node_to_idx[observation["trucks"][active_truck]["current_node"]]
        # ROUTE menu-select (multi-convoy shared-edge): allowed_nodes are route ids, scored directly.
        mask_idxs = (list(allowed_nodes) if getattr(self.actor, "menu_routes", None) is not None
                     else [node_to_idx[nid] for nid in allowed_nodes])
        taken = None
        if getattr(self.actor, "menu_routes", None) is not None:
            rc = observation.get("routed_convoys", [])
            taken = torch.tensor([float(rc.count(r)) for r in allowed_nodes], device=self.device)

        # 2. Get policy distribution
        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(pyg_data, active_idx, mask_idxs, taken)
        self.actor.train()

        if len(probs) == 0:
            return {}

        # 3. Sample Action
        if deterministic:
            action_idx = torch.argmax(probs).item()
        else:
            dist = torch.distributions.Categorical(probs)
            action_idx = dist.sample().item()

        chosen_node = allowed_nodes[action_idx]
        return {active_truck: chosen_node}

    def update(self, batch_size: int) -> dict[str, float] | None:
        """Update model parameters using a batch of SMDP transitions."""
        if len(self.replay_buffer) < batch_size:
            return None

        transitions = self.replay_buffer.sample(batch_size)

        critic_losses = []
        actor_losses = []
        alpha_losses = []
        alpha_foll_losses = []  # role_alpha: follower-group temperature loss (multi-convoy)
        q_values_list = []
        entropies_list = []
        q_spreads = []  # diagnostic: how discriminative the critic is across allowed actions

        # --- Pass 1: parse + featurize all valid samples, collate graphs ---
        # The GATv2 encoder is the hot path (one pass per network per sample previously).
        # We encode the whole minibatch in a single Batch.from_data_list pass, then apply
        # each network's `head` per-sample. Disjoint batching is identical to per-sample
        # (see tests/test_batched_equivalence.py).
        samples: list[dict[str, Any]] = []
        state_data_list: list[Data] = []
        next_data_list: list[Data] = []
        for trans in transitions:
            state = trans.state
            action_dict = trans.action
            next_state = trans.next_state
            action_mask_dict = trans.action_mask

            active_truck = state.get("active_truck")
            next_active_truck = next_state.get("active_truck")

            if active_truck is None or active_truck not in action_dict:
                continue

            chosen_node = action_dict[active_truck]
            allowed_nodes = action_mask_dict["protagonist"][active_truck]
            if chosen_node not in allowed_nodes:
                continue
            action_idx = allowed_nodes.index(chosen_node)

            node_to_idx = node_index_map(state)  # MUST match featurize_state's row order
            active_idx = node_to_idx[state["trucks"][active_truck]["current_node"]]
            menu = getattr(self.actor, "menu_routes", None) is not None
            mask_idxs = (list(allowed_nodes) if menu
                         else [node_to_idx[nid] for nid in allowed_nodes])
            taken = [float(state.get("routed_convoys", []).count(r)) for r in allowed_nodes] if menu else None

            # Does this sample bootstrap a next-state value V(s')?
            next_slot = None
            next_active_idx = None
            next_mask_idxs = None
            next_taken = None
            if (not trans.done and next_active_truck is not None
                    and "protagonist" in next_state.get("allowed_destinations", {})):
                next_allowed = next_state["allowed_destinations"]["protagonist"].get(next_active_truck, [])
                if next_allowed:
                    next_node_to_idx = node_index_map(next_state)
                    next_active_idx = next_node_to_idx[next_state["trucks"][next_active_truck]["current_node"]]
                    next_mask_idxs = (list(next_allowed) if menu
                                      else [next_node_to_idx[nid] for nid in next_allowed])
                    next_taken = ([float(next_state.get("routed_convoys", []).count(r)) for r in next_allowed]
                                  if menu else None)
                    next_slot = len(next_data_list)
                    next_data_list.append(_cached_featurize(
                        trans, "next",
                        lambda: featurize_state(next_state, next_active_truck).to(self.device)))

            state_data_list.append(_cached_featurize(
                trans, "state",
                lambda: featurize_state(state, active_truck).to(self.device)))
            samples.append({
                "action_idx": action_idx,
                "active_idx": active_idx,
                "mask_idxs": mask_idxs,
                "allowed_len": len(allowed_nodes),
                "reward": trans.reward,
                "done": trans.done,
                "dt": trans.elapsed_ticks,
                "next_slot": next_slot,
                "next_active_idx": next_active_idx,
                "next_mask_idxs": next_mask_idxs,
                # optional per-decision entropy target carried on the state (multi-convoy
                # role-dependent entropy). None for every historical transition -> the fallback below.
                "target_entropy": state.get("target_entropy"),
                # optional per-decision temperature group (0 = default/leader, 1 = follower).
                "alpha_group": state.get("alpha_group", 0),
                # the NEXT decision's group, for the soft value V(s') (a follower successor state
                # must use the follower temperature in its entropy term, not the leader's; the
                # 2026-07-09 role-alpha target fix). 0 for every historical transition.
                "next_alpha_group": next_state.get("alpha_group", 0),
                "taken": taken, "next_taken": next_taken,  # menu route-correlation (lever 2)
            })

        if not samples:
            return None

        # --- Batch-encode states once per network (and next-states for targets) ---
        sx, sei, sea, s_ptr = _collate_graphs(state_data_list)
        sx = _clip_x(sx, self.node_in_dim)
        sea = _clip_ea(sea, self.edge_in_dim)
        h_actor = self.actor.encoder(sx, sei, sea)
        h_q1 = self.q1.encoder(sx, sei, sea)
        h_q2 = self.q2.encoder(sx, sei, sea)

        n_ptr = None
        if next_data_list:
            nx, nei, nea, n_ptr = _collate_graphs(next_data_list)
            nx = _clip_x(nx, self.node_in_dim)
            nea = _clip_ea(nea, self.edge_in_dim)
            with torch.no_grad():
                h_actor_next = self.actor.encoder(nx, nei, nea)
                h_tq1 = self.target_q1.encoder(nx, nei, nea)
                h_tq2 = self.target_q2.encoder(nx, nei, nea)

        # --- Pass 2: per-sample heads on precomputed embeddings ---
        for i, s in enumerate(samples):
            hs = slice(s_ptr[i], s_ptr[i + 1])
            active_idx = s["active_idx"]
            mask_idxs = s["mask_idxs"]
            action_idx = s["action_idx"]
            tk = torch.tensor(s["taken"], device=self.device) if s["taken"] is not None else None

            # 1. Target value V(s')
            with torch.no_grad():
                if s["next_slot"] is None:
                    v_next = torch.tensor(0.0, device=self.device)
                else:
                    j = s["next_slot"]
                    hn = slice(n_ptr[j], n_ptr[j + 1])
                    nt = torch.tensor(s["next_taken"], device=self.device) if s["next_taken"] is not None else None
                    next_probs, _ = self.actor.head(h_actor_next[hn], s["next_active_idx"], s["next_mask_idxs"], nt)
                    q1_target = self.target_q1.head(h_tq1[hn], s["next_active_idx"], s["next_mask_idxs"], nt)
                    q2_target = self.target_q2.head(h_tq2[hn], s["next_active_idx"], s["next_mask_idxs"], nt)
                    min_q_target = torch.min(q1_target, q2_target)
                    log_next_probs = torch.log(next_probs + 1e-9)
                    # role_alpha: the entropy term of V(s') uses the temperature of the decision
                    # taken AT s' (follower successors get the follower alpha), unless the
                    # legacy_next_alpha isolation flag reverts to the pre-fix behaviour.
                    a_next = (self.alpha_foll
                              if (self.role_alpha and not self.legacy_next_alpha
                                  and s.get("next_alpha_group", 0) == 1)
                              else self.alpha)
                    v_next = torch.sum(next_probs * (min_q_target - a_next * log_next_probs))

            # SMDP Bellman target: y = r + gamma^dt * (1 - done) * V(s')
            target_q = (s["reward"] * self.reward_scale) + (self.gamma ** s["dt"]) * (1.0 - float(s["done"])) * v_next

            # 2. Current Q estimation
            q1_val = self.q1.head(h_q1[hs], active_idx, mask_idxs, tk)[action_idx]
            q2_val = self.q2.head(h_q2[hs], active_idx, mask_idxs, tk)[action_idx]
            q_values_list.append(torch.min(q1_val, q2_val).item())
            critic_loss = F.mse_loss(q1_val, target_q.detach()) + F.mse_loss(q2_val, target_q.detach())
            critic_losses.append(critic_loss)

            # 3. Actor loss
            probs, _ = self.actor.head(h_actor[hs], active_idx, mask_idxs, tk)
            log_probs = torch.log(probs + 1e-9)
            curr_q1 = self.q1.head(h_q1[hs], active_idx, mask_idxs, tk)
            curr_q2 = self.q2.head(h_q2[hs], active_idx, mask_idxs, tk)
            # CRITICAL: Detach critic predictions to block policy gradients from critic networks
            min_q = torch.min(curr_q1, curr_q2).detach()
            if min_q.numel() > 1:
                # Q-spread across allowed destinations: ~0 means the critic can't tell good routes
                # from bad ones (the protagonist's reward signal-to-noise failure mode).
                q_spreads.append((min_q.max() - min_q.min()).item())
            # role_alpha: followers (alpha_group==1) use the follower temperature in the actor loss.
            use_foll = self.role_alpha and s.get("alpha_group", 0) == 1
            a_val = self.alpha_foll if use_foll else self.alpha
            actor_loss = torch.sum(probs * (a_val * log_probs - min_q))
            actor_losses.append(actor_loss)
            entropy = -torch.sum(probs * log_probs)
            entropies_list.append(entropy.item())

            # 4. Alpha entropy tuning loss
            if self.autotune_alpha:
                per_sample_te = s.get("target_entropy")
                if per_sample_te is not None:
                    # per-decision target (multi-convoy role-dependent entropy: leader high,
                    # followers ~0). Absent for every historical transition -> falls back below.
                    target_entropy = torch.tensor(float(per_sample_te), device=self.device)
                elif self.target_entropy is not None:
                    target_entropy = torch.tensor(self.target_entropy, device=self.device)
                else:
                    # Dynamic target entropy based on number of active valid actions (calibrated to 0.45)
                    target_entropy = -0.45 * torch.log(torch.tensor(1.0 / s["allowed_len"], device=self.device))
                # Negative feedback: alpha decreases when entropy > target (standard SAC temperature loss).
                # The previous `-log_alpha * (...)` had the sign inverted -> alpha runaway / critic divergence.
                if use_foll:
                    alpha_foll_losses.append(self.log_alpha_foll * (entropy - target_entropy).detach())
                else:
                    alpha_losses.append(self.log_alpha * (entropy - target_entropy).detach())

        if not critic_losses:
            return None

        # --- 5. BACKPROPAGATION ---
        total_critic_loss = torch.stack(critic_losses).mean()
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)

        critic_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q1.parameters() if p.grad is not None)
        critic_grad_norm += sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q2.parameters() if p.grad is not None)
        critic_grad_norm = critic_grad_norm ** 0.5
        
        self.critic_optimizer.step()

        total_actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)

        actor_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.actor.parameters() if p.grad is not None)
        actor_grad_norm = actor_grad_norm ** 0.5
        
        self.actor_optimizer.step()

        if self.autotune_alpha and alpha_losses:
            total_alpha_loss = torch.stack(alpha_losses).mean()
            self.alpha_optimizer.zero_grad()
            total_alpha_loss.backward()
            self.alpha_optimizer.step()
            if self.alpha_floor is not None:
                with torch.no_grad():
                    self.log_alpha.clamp_(min=math.log(self.alpha_floor))
            self.alpha = self.log_alpha.exp().item()
        else:
            total_alpha_loss = torch.tensor(0.0)
        if self.autotune_alpha and self.role_alpha and alpha_foll_losses:
            foll_loss = torch.stack(alpha_foll_losses).mean()
            self.alpha_foll_optimizer.zero_grad()
            foll_loss.backward()
            self.alpha_foll_optimizer.step()
            self.alpha_foll = self.log_alpha_foll.exp().item()

        # Soft update target critics
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "protag_critic_loss": total_critic_loss.item(),
            "protag_actor_loss": total_actor_loss.item(),
            "protag_alpha_loss": total_alpha_loss.item(),
            "protag_alpha": self.alpha,
            "protag_alpha_foll": getattr(self, "alpha_foll", self.alpha),
            "protag_q_val": sum(q_values_list) / len(q_values_list) if q_values_list else 0.0,
            "protag_entropy": sum(entropies_list) / len(entropies_list) if entropies_list else 0.0,
            "protag_q_spread": sum(q_spreads) / len(q_spreads) if q_spreads else 0.0,
            "protag_critic_grad_norm": critic_grad_norm,
            "protag_actor_grad_norm": actor_grad_norm,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str, episode: int = 0) -> None:
        """Save a complete mathematical checkpoint to disk."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "episode": episode,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
        }
        if self.autotune_alpha:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load a complete mathematical checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        self.replay_buffer.buffer.clear()
        self.replay_buffer.buffer.extend(checkpoint["replay_buffer"])
        
        if self.autotune_alpha and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].data)
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            self.alpha = self.log_alpha.exp().item()
            
        return checkpoint.get("episode", 0)


class AntagonistQNet(nn.Module):
    """Twin Q-network implementation for the Antagonist (Adversary)."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        num_congestion_levels: int = 4,
    ) -> None:
        super().__init__()
        from src.agents.networks import GATv2Encoder

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

    def forward(
        self,
        pyg_data: Data,
        original_edges: list[tuple[Any, Any]],
        node_to_idx: Mapping[Any, int],
        allowed_edges: list[tuple[Any, Any]],
        remaining_budget: float,
        level_costs: list[float],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate state-action Q-values for edge choice and level choices."""
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Q-head on precomputed node embeddings (see AntagonistPolicyValueNet.head)."""
        device = h_nodes.device
        num_allowed = len(allowed_edges)

        h_graph = torch.mean(h_nodes, dim=0, keepdim=True)

        if num_allowed == 0:
            wait_q = self.wait_scorer(h_graph).squeeze(-1)  # Only Wait action Q-value [1]
            level_q = torch.empty((0, len(level_costs)), device=device)
            return wait_q, level_q

        # Build edge encodings
        h_edges = []
        u_list = edge_index[0].tolist()
        v_list = edge_index[1].tolist()
        edge_features_dict = dict(zip(zip(u_list, v_list), edge_attr))

        for u, v in allowed_edges:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            emb_u = h_nodes[idx_u]
            emb_v = h_nodes[idx_v]
            attr = edge_features_dict.get((idx_u, idx_v), torch.zeros(edge_attr.size(1), device=device))
            combined = torch.cat([emb_u, emb_v, attr], dim=-1)
            h_edges.append(combined)

        h_edges_tensor = torch.stack(h_edges, dim=0)
        h_edge_features = self.edge_mlp(h_edges_tensor)

        # Compute Q components
        edge_q = self.edge_scorer(h_edge_features).squeeze(-1)  # [num_allowed]
        wait_q = self.wait_scorer(h_graph).squeeze(-1)  # [1]
        all_edge_q = torch.cat([edge_q, wait_q], dim=0)  # [num_allowed + 1]

        level_q = self.level_head(h_edge_features)  # [num_allowed, num_levels]

        # Action masking for costs exceeding budget
        level_mask = torch.tensor(
            [cost <= remaining_budget + 1e-6 for cost in level_costs],
            dtype=torch.bool,
            device=device,
        )
        level_mask_expanded = level_mask.unsqueeze(0).expand(num_allowed, -1)
        level_q = level_q.masked_fill(~level_mask_expanded, -1e9)

        return all_edge_q, level_q


class AntagonistSAC:
    """Soft Actor-Critic agent for the adversarial Antagonist."""

    def __init__(
        self,
        node_in_dim: int = 13,
        edge_in_dim: int = 4,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        num_congestion_levels: int = 4,
        level_costs: list[float] | None = None,
        congestion_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
        lr_actor: float = 3e-4,
        lr_critic: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 1.0,
        autotune_alpha: bool = True,
        buffer_capacity: int = 50000,
        device: str | None = None,
        reward_scale: float = 0.001,
        target_entropy: float | None = None,
    ) -> None:
        self.device = device or get_torch_device()
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.target_entropy = target_entropy
        self.num_levels = num_congestion_levels
        # The actual congestion-level VALUES the level-head index maps to. Must match the env's
        # SMDPConfig.congestion_levels, else select_action returns a level the action mask rejects
        # (e.g. full-blockage-only configs). Was previously hardcoded to [0.25,0.5,0.75,1.0].
        self.congestion_levels = tuple(congestion_levels)
        # Level cost calculation defaults based on: level * duration * congestion_cost
        # E.g. levels = [0.25, 0.50, 0.75, 1.0], duration = 12, cost = 0.015
        self.level_costs = level_costs or [
            level * 12 * 0.015 for level in [0.25, 0.5, 0.75, 1.0]
        ]
        # Feature widths this agent's networks consume (see ProtagonistSAC / infer_*_in_dim).
        self.node_in_dim = node_in_dim
        self.edge_in_dim = edge_in_dim

        # 1. Initialize networks
        self.actor = AntagonistPolicyValueNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_congestion_levels=num_congestion_levels,
        ).to(self.device)

        self.q1 = AntagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_congestion_levels=num_congestion_levels,
        ).to(self.device)

        self.q2 = AntagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_congestion_levels=num_congestion_levels,
        ).to(self.device)

        # 2. Target Critics
        self.target_q1 = AntagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_congestion_levels=num_congestion_levels,
        ).to(self.device)
        self.target_q1.load_state_dict(self.q1.state_dict())

        self.target_q2 = AntagonistQNet(
            node_in_dim=node_in_dim,
            edge_in_dim=edge_in_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            num_congestion_levels=num_congestion_levels,
        ).to(self.device)
        self.target_q2.load_state_dict(self.q2.state_dict())

        # 3. Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=lr_critic,
        )

        # 4. Temperature Entropy alpha
        self.autotune_alpha = autotune_alpha
        if autotune_alpha:
            self.log_alpha = torch.tensor(
                math.log(alpha_init),
                requires_grad=True,
                device=self.device,
                dtype=torch.float32,
            )
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_actor)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        # 5. Experience Replay Buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

    def select_action(
        self,
        observation: dict[str, Any],
        action_mask: dict[str, Any],
        remaining_budget: float,
        deterministic: bool = False,
    ) -> tuple[tuple[Any, Any], float] | None:
        """Select a congestion action (edge, level) or None (wait) using the policy.

        Returns
        -------
        tuple[tuple[Any, Any], float] | None:
            A choice of `((u, v), level)` or `None`.
        """
        allowed_edges = action_mask.get("allowed_edges")
        if allowed_edges is None:
            allowed_edges = list(action_mask.get("levels_by_edge", {}).keys())

        original_edges = action_mask.get("original_edges")
        if original_edges is None:
            original_edges = list(observation.get("edges", {}).keys())

        # 1. Featurize state and maps
        pyg_data = featurize_state(observation).to(self.device)
        pyg_data.x = _clip_x(pyg_data.x, self.node_in_dim)
        pyg_data.edge_attr = _clip_ea(pyg_data.edge_attr, self.edge_in_dim)
        node_to_idx = node_index_map(observation)  # MUST match featurize_state's row order

        # 2. Get actor policy distributions
        self.actor.eval()
        with torch.no_grad():
            edge_probs, level_probs, _ = self.actor(
                pyg_data,
                original_edges,
                node_to_idx,
                allowed_edges,
                remaining_budget,
                self.level_costs,
            )
        self.actor.train()

        num_allowed = len(allowed_edges)

        # 3. Joint action space flattening & selection with cost budget masking
        # Size: M * L + 1
        flat_probs = torch.zeros(num_allowed * self.num_levels + 1, device=self.device)
        if num_allowed > 0:
            edge_p = edge_probs[:-1].unsqueeze(1)
            joint_p = (edge_p * level_probs).reshape(-1)
            costs = torch.tensor(self.level_costs, device=self.device)
            mask = (costs <= remaining_budget + 1e-6).unsqueeze(0).expand(num_allowed, -1).reshape(-1)
            joint_p = torch.where(mask, joint_p, torch.zeros_like(joint_p))
            flat_probs[:-1] = joint_p
        flat_probs[-1] = edge_probs[-1]  # Wait action (cost = 0, always allowed)

        sum_probs = torch.sum(flat_probs)
        if sum_probs < 1e-8:
            return None

        # Normalize to prevent floating point inaccuracies
        flat_probs = flat_probs / sum_probs

        if deterministic:
            chosen_idx = torch.argmax(flat_probs).item()
        else:
            dist = torch.distributions.Categorical(flat_probs)
            chosen_idx = dist.sample().item()

        # Wait action selected
        if chosen_idx == num_allowed * self.num_levels:
            return None

        chosen_edge_idx = chosen_idx // self.num_levels
        chosen_level_idx = chosen_idx % self.num_levels

        chosen_edge = allowed_edges[chosen_edge_idx]
        chosen_level = self.congestion_levels[chosen_level_idx]

        return chosen_edge, chosen_level

    def update(self, batch_size: int) -> dict[str, float] | None:
        """Update model parameters using a batch of antagonist SMDP transitions."""
        if len(self.replay_buffer) < batch_size:
            return None

        transitions = self.replay_buffer.sample(batch_size)

        critic_losses = []
        actor_losses = []
        alpha_losses = []
        q_values_list = []
        entropies_list = []

        # --- Pass 1: parse + featurize all valid samples, collate graphs ---
        # See ProtagonistSAC.update for the batch-encode rationale; the antagonist already
        # computes each network's output once per sample, so the per-sample head logic below
        # is reused verbatim on precomputed embeddings.
        samples: list[dict[str, Any]] = []
        state_data_list: list[Data] = []
        next_data_list: list[Data] = []
        for trans in transitions:
            state = trans.state
            action = trans.action  # ((u, v), level) or None
            next_state = trans.next_state

            action_mask_dict = trans.action_mask.get("antagonist", {})
            allowed_edges = action_mask_dict.get("allowed_edges")
            if allowed_edges is None:
                allowed_edges = list(action_mask_dict.get("levels_by_edge", {}).keys())
            remaining_budget = trans.info.get("antagonist_budget_remaining", 0.0)

            node_to_idx = node_index_map(state)  # MUST match featurize_state's row order
            num_allowed = len(allowed_edges)

            # Resolve chosen action index in the flattened (edge x level + wait) space
            if action is None:
                chosen_flat_idx = num_allowed * self.num_levels
            else:
                edge, level = action
                reversed_edge = (edge[1], edge[0])
                if edge in allowed_edges:
                    edge_idx = allowed_edges.index(edge)
                elif reversed_edge in allowed_edges:
                    edge_idx = allowed_edges.index(reversed_edge)
                else:
                    continue  # Invalid choice
                # Map the level VALUE back to its index in this agent's congestion_levels (was
                # hardcoded [0.25,0.5,0.75,1.0] -> broke any non-default level set, e.g. (1.0,)).
                if level in self.congestion_levels:
                    level_idx = self.congestion_levels.index(level)
                else:
                    level_idx = min(range(self.num_levels),
                                    key=lambda k: abs(self.congestion_levels[k] - level))
                chosen_flat_idx = edge_idx * self.num_levels + level_idx

            rec: dict[str, Any] = {
                "allowed_edges": allowed_edges,
                "node_to_idx": node_to_idx,
                "num_allowed": num_allowed,
                "remaining_budget": remaining_budget,
                "chosen_flat_idx": chosen_flat_idx,
                "reward": trans.reward,
                "done": trans.done,
                "dt": trans.elapsed_ticks,
                "next_slot": None,
            }

            if not trans.done:
                next_mask_dict = next_state.get("allowed_destinations", {}).get("antagonist", {})
                next_allowed = next_mask_dict.get("allowed_edges")
                if next_allowed is None:
                    next_allowed = list(next_mask_dict.get("levels_by_edge", {}).keys())
                next_budget = trans.info.get("next_antagonist_budget_remaining", remaining_budget)
                next_node_to_idx = node_index_map(next_state)
                rec["next_allowed"] = next_allowed
                rec["next_node_to_idx"] = next_node_to_idx
                rec["next_budget"] = next_budget
                rec["next_num_allowed"] = len(next_allowed)
                rec["next_slot"] = len(next_data_list)
                next_data_list.append(_cached_featurize(
                    trans, "next", lambda: featurize_state(next_state).to(self.device)))

            state_data_list.append(_cached_featurize(
                trans, "state", lambda: featurize_state(state).to(self.device)))
            samples.append(rec)

        if not samples:
            return None

        # --- Batch-encode states once per network (and next-states for targets) ---
        sx, sei, sea, s_ptr = _collate_graphs(state_data_list)
        sx = _clip_x(sx, self.node_in_dim)
        sea = _clip_ea(sea, self.edge_in_dim)
        h_actor = self.actor.encoder(sx, sei, sea)
        h_q1 = self.q1.encoder(sx, sei, sea)
        h_q2 = self.q2.encoder(sx, sei, sea)

        n_ptr = None
        if next_data_list:
            ncx, ncei, ncea, n_ptr = _collate_graphs(next_data_list)
            ncx = _clip_x(ncx, self.node_in_dim)
            ncea = _clip_ea(ncea, self.edge_in_dim)
            with torch.no_grad():
                h_actor_next = self.actor.encoder(ncx, ncei, ncea)
                h_tq1 = self.target_q1.encoder(ncx, ncei, ncea)
                h_tq2 = self.target_q2.encoder(ncx, ncei, ncea)

        # --- Pass 2: per-sample heads on precomputed embeddings ---
        for i, rec in enumerate(samples):
            hs = slice(s_ptr[i], s_ptr[i + 1])
            ei_s = state_data_list[i].edge_index
            ea_s = _clip_ea(state_data_list[i].edge_attr, self.edge_in_dim)
            allowed_edges = rec["allowed_edges"]
            node_to_idx = rec["node_to_idx"]
            num_allowed = rec["num_allowed"]
            remaining_budget = rec["remaining_budget"]
            chosen_flat_idx = rec["chosen_flat_idx"]
            reward = rec["reward"]
            done = rec["done"]
            dt = rec["dt"]

            # --- 2. CRITIC LOSS: TARGET VALUE ESTIMATION V(s') ---
            with torch.no_grad():
                if done:
                    v_next = torch.tensor(0.0, device=self.device)
                else:
                    j = rec["next_slot"]
                    hn = slice(n_ptr[j], n_ptr[j + 1])
                    nei = next_data_list[j].edge_index
                    nea = _clip_ea(next_data_list[j].edge_attr, self.edge_in_dim)
                    next_allowed = rec["next_allowed"]
                    next_node_to_idx = rec["next_node_to_idx"]
                    next_budget = rec["next_budget"]
                    next_num_allowed = rec["next_num_allowed"]

                    if next_num_allowed == 0:
                        # Only wait action is allowed
                        q1_target_wait, _ = self.target_q1.head(
                            h_tq1[hn], nei, nea, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        q2_target_wait, _ = self.target_q2.head(
                            h_tq2[hn], nei, nea, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        min_q_wait = torch.min(q1_target_wait[0], q2_target_wait[0])
                        v_next = min_q_wait  # log pi(wait|s) = log(1.0) = 0
                    else:
                        next_edge_probs, next_level_probs, _ = self.actor.head(
                            h_actor_next[hn], nei, nea, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        next_q1_edge, next_q1_level = self.target_q1.head(
                            h_tq1[hn], nei, nea, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        next_q2_edge, next_q2_level = self.target_q2.head(
                            h_tq2[hn], nei, nea, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )

                        # Flatten next Q and probabilities with cost budget masking
                        next_flat_probs = torch.zeros(next_num_allowed * self.num_levels + 1, device=self.device)
                        next_flat_q1 = torch.zeros_like(next_flat_probs)
                        next_flat_q2 = torch.zeros_like(next_flat_probs)

                        if next_num_allowed > 0:
                            edge_p = next_edge_probs[:-1].unsqueeze(1)
                            joint_p = (edge_p * next_level_probs).reshape(-1)
                            
                            q1_e = next_q1_edge[:-1].unsqueeze(1)
                            joint_q1 = (q1_e + next_q1_level).reshape(-1)
                            
                            q2_e = next_q2_edge[:-1].unsqueeze(1)
                            joint_q2 = (q2_e + next_q2_level).reshape(-1)
                            
                            costs = torch.tensor(self.level_costs, device=self.device)
                            mask = (costs <= next_budget + 1e-6).unsqueeze(0).expand(next_num_allowed, -1).reshape(-1)
                            
                            joint_p = torch.where(mask, joint_p, torch.zeros_like(joint_p))
                            joint_q1 = torch.where(mask, joint_q1, torch.full_like(joint_q1, -1e9))
                            joint_q2 = torch.where(mask, joint_q2, torch.full_like(joint_q2, -1e9))
                            
                            next_flat_probs[:-1] = joint_p
                            next_flat_q1[:-1] = joint_q1
                            next_flat_q2[:-1] = joint_q2
                        
                        next_flat_probs[-1] = next_edge_probs[-1]
                        next_flat_q1[-1] = next_q1_edge[-1]
                        next_flat_q2[-1] = next_q2_edge[-1]

                        # Prevent div by 0 and normalize
                        sum_probs = torch.sum(next_flat_probs)
                        if sum_probs < 1e-8:
                            next_flat_probs.zero_()
                            next_flat_probs[-1] = 1.0
                            sum_probs = 1.0
                        next_flat_probs = next_flat_probs / sum_probs
                        min_q_next = torch.min(next_flat_q1, next_flat_q2)

                        log_next_probs = torch.log(next_flat_probs + 1e-9)
                        v_next = torch.sum(next_flat_probs * (min_q_next - self.alpha * log_next_probs))

            target_q = (reward * self.reward_scale) + (self.gamma ** dt) * (1.0 - float(done)) * v_next

            # --- 3. CRITIC LOSS: CURRENT Q ESTIMATION ---
            q1_edge, q1_level = self.q1.head(
                h_q1[hs], ei_s, ea_s, node_to_idx, allowed_edges, remaining_budget, self.level_costs
            )
            q2_edge, q2_level = self.q2.head(
                h_q2[hs], ei_s, ea_s, node_to_idx, allowed_edges, remaining_budget, self.level_costs
            )

            # Reconstruct chosen Q-value
            if chosen_flat_idx == num_allowed * self.num_levels:
                q1_val = q1_edge[-1]
                q2_val = q2_edge[-1]
            else:
                edge_idx = chosen_flat_idx // self.num_levels
                level_idx = chosen_flat_idx % self.num_levels
                q1_val = q1_edge[edge_idx] + q1_level[edge_idx, level_idx]
                q2_val = q2_edge[edge_idx] + q2_level[edge_idx, level_idx]
                
            q_values_list.append(torch.min(q1_val, q2_val).item())

            critic_loss = F.mse_loss(q1_val, target_q.detach()) + F.mse_loss(q2_val, target_q.detach())
            critic_losses.append(critic_loss)

            # --- 4. ACTOR LOSS & ALPHA ENTROPY TUNING ---
            edge_probs, level_probs, _ = self.actor.head(
                h_actor[hs], ei_s, ea_s, node_to_idx, allowed_edges, remaining_budget, self.level_costs
            )

            if num_allowed > 0:
                flat_probs = torch.zeros(num_allowed * self.num_levels + 1, device=self.device)
                flat_q1 = torch.zeros_like(flat_probs)
                flat_q2 = torch.zeros_like(flat_probs)

                # Flatten actor representations with cost budget masking
                if num_allowed > 0:
                    edge_p = edge_probs[:-1].unsqueeze(1)
                    joint_p = (edge_p * level_probs).reshape(-1)
                    
                    q1_e = q1_edge[:-1].unsqueeze(1)
                    joint_q1 = (q1_e + q1_level).reshape(-1)
                    
                    q2_e = q2_edge[:-1].unsqueeze(1)
                    joint_q2 = (q2_e + q2_level).reshape(-1)
                    
                    costs = torch.tensor(self.level_costs, device=self.device)
                    mask = (costs <= remaining_budget + 1e-6).unsqueeze(0).expand(num_allowed, -1).reshape(-1)
                    
                    joint_p = torch.where(mask, joint_p, torch.zeros_like(joint_p))
                    joint_q1 = torch.where(mask, joint_q1, torch.full_like(joint_q1, -1e9))
                    joint_q2 = torch.where(mask, joint_q2, torch.full_like(joint_q2, -1e9))
                    
                    flat_probs[:-1] = joint_p
                    flat_q1[:-1] = joint_q1
                    flat_q2[:-1] = joint_q2

                flat_probs[-1] = edge_probs[-1]
                flat_q1[-1] = q1_edge[-1]
                flat_q2[-1] = q2_edge[-1]

                sum_probs = torch.sum(flat_probs)
                if sum_probs < 1e-8:
                    flat_probs.zero_()
                    flat_probs[-1] = 1.0
                    sum_probs = 1.0
                
                flat_probs = flat_probs / sum_probs
                
                # CRITICAL: Detach critic predictions to block policy gradients from critic networks
                min_q = torch.min(flat_q1, flat_q2).detach()

                log_probs = torch.log(flat_probs + 1e-9)
                
                entropy = -torch.sum(flat_probs * log_probs)
                entropies_list.append(entropy.item())

                actor_loss = torch.sum(flat_probs * (self.alpha * log_probs - min_q))
                actor_losses.append(actor_loss)

                if self.autotune_alpha:
                    if self.target_entropy is not None:
                        target_entropy = torch.tensor(self.target_entropy, device=self.device)
                    else:
                        # Dynamic target entropy over flat options size.
                        # protag_signal_rebalance: 0.98 (near-max) forced near-uniform play, so alpha
                        # climbed unboundedly to fight a policy that wanted to commit -> 0.5 lets it settle.
                        target_entropy = -0.5 * torch.log(torch.tensor(1.0 / len(flat_probs), device=self.device))
                    # Negative feedback: alpha decreases when entropy > target (standard SAC temperature loss).
                    # The previous `-log_alpha * (...)` had the sign inverted -> alpha runaway / critic divergence.
                    alpha_loss = self.log_alpha * (entropy - target_entropy).detach()
                    alpha_losses.append(alpha_loss)

        if not critic_losses:
            return None

        # --- 5. BACKPROPAGATION ---
        total_critic_loss = torch.stack(critic_losses).mean()
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.q1.parameters()) + list(self.q2.parameters()), 10.0)

        critic_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q1.parameters() if p.grad is not None)
        critic_grad_norm += sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q2.parameters() if p.grad is not None)
        critic_grad_norm = critic_grad_norm ** 0.5
        
        self.critic_optimizer.step()

        if actor_losses:
            total_actor_loss = torch.stack(actor_losses).mean()
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10.0)

            actor_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.actor.parameters() if p.grad is not None)
            actor_grad_norm = actor_grad_norm ** 0.5
            
            self.actor_optimizer.step()
        else:
            total_actor_loss = torch.tensor(0.0, device=self.device)
            actor_grad_norm = 0.0

        if self.autotune_alpha and alpha_losses:
            total_alpha_loss = torch.stack(alpha_losses).mean()
            self.alpha_optimizer.zero_grad()
            total_alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            total_alpha_loss = torch.tensor(0.0, device=self.device)

        # Soft update target critics
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "antag_critic_loss": total_critic_loss.item(),
            "antag_actor_loss": total_actor_loss.item(),
            "antag_alpha_loss": total_alpha_loss.item(),
            "antag_alpha": self.alpha,
            "antag_q_val": sum(q_values_list) / len(q_values_list) if q_values_list else 0.0,
            "antag_entropy": sum(entropies_list) / len(entropies_list) if entropies_list else 0.0,
            "antag_critic_grad_norm": critic_grad_norm,
            "antag_actor_grad_norm": actor_grad_norm,
        }

    def _soft_update(self, source: nn.Module, target: nn.Module) -> None:
        for source_param, target_param in zip(source.parameters(), target.parameters()):
            target_param.data.copy_(
                self.tau * source_param.data + (1.0 - self.tau) * target_param.data
            )

    def save_checkpoint(self, filepath: str, episode: int = 0) -> None:
        """Save a complete mathematical checkpoint to disk."""
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        checkpoint = {
            "episode": episode,
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "target_q1": self.target_q1.state_dict(),
            "target_q2": self.target_q2.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "replay_buffer": list(self.replay_buffer.buffer),
        }
        if self.autotune_alpha:
            checkpoint["log_alpha"] = self.log_alpha
            checkpoint["alpha_optimizer"] = self.alpha_optimizer.state_dict()
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, filepath: str) -> int:
        """Load a complete mathematical checkpoint from disk."""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.q1.load_state_dict(checkpoint["q1"])
        self.q2.load_state_dict(checkpoint["q2"])
        self.target_q1.load_state_dict(checkpoint["target_q1"])
        self.target_q2.load_state_dict(checkpoint["target_q2"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        
        self.replay_buffer.buffer.clear()
        self.replay_buffer.buffer.extend(checkpoint["replay_buffer"])
        
        if self.autotune_alpha and "log_alpha" in checkpoint:
            self.log_alpha.data.copy_(checkpoint["log_alpha"].data)
            self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
            self.alpha = self.log_alpha.exp().item()
            
        return checkpoint.get("episode", 0)
