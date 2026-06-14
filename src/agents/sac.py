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
from torch_geometric.data import Data

from src.agents.device import get_torch_device
from src.agents.networks import (
    AntagonistPolicyValueNet,
    ProtagonistPolicyValueNet,
    featurize_state,
)


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
        node_in_dim: int = 7,
        edge_in_dim: int = 2,
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
        if not action_mask_indices:
            return torch.empty(0, device=h.device)

        h_active = h[active_node_idx].unsqueeze(0)
        h_candidates = h[action_mask_indices]
        h_active_rep = h_active.expand(h_candidates.size(0), -1)

        q_values = self.q_bilinear(h_active_rep, h_candidates).squeeze(-1)  # [num_candidates]
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
    ) -> None:
        self.device = device or get_torch_device()
        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale
        self.target_entropy = target_entropy

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
        node_ids = list(observation["nodes"].keys())
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

        active_idx = node_to_idx[observation["trucks"][active_truck]["current_node"]]
        mask_idxs = [node_to_idx[nid] for nid in allowed_nodes]

        # 2. Get policy distribution
        self.actor.eval()
        with torch.no_grad():
            probs, _ = self.actor(pyg_data, active_idx, mask_idxs)
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
        q_values_list = []
        entropies_list = []

        for trans in transitions:
            # Parse transition components
            state = trans.state
            action_dict = trans.action
            reward = trans.reward
            next_state = trans.next_state
            done = trans.done
            dt = trans.elapsed_ticks
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

            # Node indices map
            node_ids = list(state["nodes"].keys())
            node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

            active_idx = node_to_idx[state["trucks"][active_truck]["current_node"]]
            mask_idxs = [node_to_idx[nid] for nid in allowed_nodes]

            pyg_data = featurize_state(state, active_truck).to(self.device)

            # --- 1. CRITIC LOSS: TARGET VALUE ESTIMATION V(s') ---
            with torch.no_grad():
                if done:
                    v_next = torch.tensor(0.0, device=self.device)
                elif next_active_truck is None or "protagonist" not in next_state.get("allowed_destinations", {}):
                    v_next = torch.tensor(0.0, device=self.device)
                else:
                    next_allowed = next_state["allowed_destinations"]["protagonist"].get(next_active_truck, [])
                    if not next_allowed:
                        v_next = torch.tensor(0.0, device=self.device)
                    else:
                        next_node_ids = list(next_state["nodes"].keys())
                        next_node_to_idx = {nid: idx for idx, nid in enumerate(next_node_ids)}
                        next_active_idx = next_node_to_idx[next_state["trucks"][next_active_truck]["current_node"]]
                        next_mask_idxs = [next_node_to_idx[nid] for nid in next_allowed]

                        next_pyg_data = featurize_state(next_state, next_active_truck).to(self.device)

                        # Policy probs next: pi(a' | s')
                        next_probs, _ = self.actor(next_pyg_data, next_active_idx, next_mask_idxs)
                        
                        # Target Critics next
                        q1_target = self.target_q1(next_pyg_data, next_active_idx, next_mask_idxs)
                        q2_target = self.target_q2(next_pyg_data, next_active_idx, next_mask_idxs)
                        min_q_target = torch.min(q1_target, q2_target)

                        # V(s') = sum_a' pi(a'|s') * [min Q(s', a') - alpha * log pi(a'|s')]
                        log_next_probs = torch.log(next_probs + 1e-9)
                        v_next = torch.sum(next_probs * (min_q_target - self.alpha * log_next_probs))

            # SMDP Bellman target: y = r + gamma^dt * (1 - done) * V(s')
            target_q = (reward * self.reward_scale) + (self.gamma ** dt) * (1.0 - float(done)) * v_next

            # --- 2. CRITIC LOSS: CURRENT Q ESTIMATION ---
            q1_val = self.q1(pyg_data, active_idx, mask_idxs)[action_idx]
            q2_val = self.q2(pyg_data, active_idx, mask_idxs)[action_idx]
            
            q_values_list.append(torch.min(q1_val, q2_val).item())

            critic_loss = F.mse_loss(q1_val, target_q.detach()) + F.mse_loss(q2_val, target_q.detach())
            critic_losses.append(critic_loss)

            # --- 3. ACTOR LOSS ---
            probs, _ = self.actor(pyg_data, active_idx, mask_idxs)
            log_probs = torch.log(probs + 1e-9)

            curr_q1 = self.q1(pyg_data, active_idx, mask_idxs)
            curr_q2 = self.q2(pyg_data, active_idx, mask_idxs)
            
            # CRITICAL: Detach critic predictions to block policy gradients from critic networks
            min_q = torch.min(curr_q1, curr_q2).detach()

            # Policy loss: sum_a pi(a|s) * [alpha * log pi(a|s) - Q(s, a)]
            actor_loss = torch.sum(probs * (self.alpha * log_probs - min_q))
            actor_losses.append(actor_loss)

            entropy = -torch.sum(probs * log_probs)
            entropies_list.append(entropy.item())

            # --- 4. ALPHA ENTROPY TUNING LOSS ---
            if self.autotune_alpha:
                if self.target_entropy is not None:
                    target_entropy = torch.tensor(self.target_entropy, device=self.device)
                else:
                    # Dynamic target entropy based on number of active valid actions (calibrated to 0.45)
                    target_entropy = -0.45 * torch.log(torch.tensor(1.0 / len(allowed_nodes), device=self.device))
                alpha_loss = -self.log_alpha * (entropy - target_entropy).detach()
                alpha_losses.append(alpha_loss)

        if not critic_losses:
            return None

        # --- 5. BACKPROPAGATION ---
        total_critic_loss = torch.stack(critic_losses).mean()
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        
        critic_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q1.parameters() if p.grad is not None)
        critic_grad_norm += sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q2.parameters() if p.grad is not None)
        critic_grad_norm = critic_grad_norm ** 0.5
        
        self.critic_optimizer.step()

        total_actor_loss = torch.stack(actor_losses).mean()
        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        
        actor_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.actor.parameters() if p.grad is not None)
        actor_grad_norm = actor_grad_norm ** 0.5
        
        self.actor_optimizer.step()

        if self.autotune_alpha and alpha_losses:
            total_alpha_loss = torch.stack(alpha_losses).mean()
            self.alpha_optimizer.zero_grad()
            total_alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        else:
            total_alpha_loss = torch.tensor(0.0)

        # Soft update target critics
        self._soft_update(self.q1, self.target_q1)
        self._soft_update(self.q2, self.target_q2)

        return {
            "protag_critic_loss": total_critic_loss.item(),
            "protag_actor_loss": total_actor_loss.item(),
            "protag_alpha_loss": total_alpha_loss.item(),
            "protag_alpha": self.alpha,
            "protag_q_val": sum(q_values_list) / len(q_values_list) if q_values_list else 0.0,
            "protag_entropy": sum(entropies_list) / len(entropies_list) if entropies_list else 0.0,
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
        node_in_dim: int = 7,
        edge_in_dim: int = 2,
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
        device = pyg_data.x.device
        num_allowed = len(allowed_edges)

        h_nodes = self.encoder(pyg_data.x, pyg_data.edge_index, pyg_data.edge_attr)
        h_graph = torch.mean(h_nodes, dim=0, keepdim=True)

        if num_allowed == 0:
            wait_q = self.wait_scorer(h_graph).squeeze(-1)  # Only Wait action Q-value [1]
            level_q = torch.empty((0, len(level_costs)), device=device)
            return wait_q, level_q

        # Build edge encodings
        h_edges = []
        u_list = pyg_data.edge_index[0].tolist()
        v_list = pyg_data.edge_index[1].tolist()
        edge_features_dict = dict(zip(zip(u_list, v_list), pyg_data.edge_attr))

        for u, v in allowed_edges:
            idx_u = node_to_idx[u]
            idx_v = node_to_idx[v]
            emb_u = h_nodes[idx_u]
            emb_v = h_nodes[idx_v]
            attr = edge_features_dict.get((idx_u, idx_v), torch.zeros(2, device=device))
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
        node_in_dim: int = 7,
        edge_in_dim: int = 2,
        hidden_dim: int = 64,
        num_layers: int = 2,
        heads: int = 4,
        num_congestion_levels: int = 4,
        level_costs: list[float] | None = None,
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
        # Level cost calculation defaults based on: level * duration * congestion_cost
        # E.g. levels = [0.25, 0.50, 0.75, 1.0], duration = 12, cost = 0.015
        self.level_costs = level_costs or [
            level * 12 * 0.015 for level in [0.25, 0.5, 0.75, 1.0]
        ]

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
        node_ids = list(observation["nodes"].keys())
        node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

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
        chosen_level = [0.25, 0.5, 0.75, 1.0][chosen_level_idx]

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

        for trans in transitions:
            # Parse transition components
            state = trans.state
            action = trans.action  # ((u, v), level) or None
            reward = trans.reward
            next_state = trans.next_state
            done = trans.done
            dt = trans.elapsed_ticks
            action_mask_dict = trans.action_mask.get("antagonist", {})

            allowed_edges = action_mask_dict.get("allowed_edges")
            if allowed_edges is None:
                allowed_edges = list(action_mask_dict.get("levels_by_edge", {}).keys())

            original_edges = action_mask_dict.get("original_edges")
            if original_edges is None:
                original_edges = list(state.get("edges", {}).keys())
            remaining_budget = trans.info.get("antagonist_budget_remaining", 0.0)

            # Node indices map
            node_ids = list(state["nodes"].keys())
            node_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

            num_allowed = len(allowed_edges)
            pyg_data = featurize_state(state).to(self.device)

            # --- 1. RESOLVE CHOSEN ACTION INDEX IN FLATTENED SPACE ---
            if action is None:
                chosen_flat_idx = num_allowed * self.num_levels
            else:
                edge, level = action
                # Support bidirectional edge lookup
                reversed_edge = (edge[1], edge[0])
                if edge in allowed_edges:
                    edge_idx = allowed_edges.index(edge)
                elif reversed_edge in allowed_edges:
                    edge_idx = allowed_edges.index(reversed_edge)
                else:
                    continue  # Invalid choice

                level_idx = [0.25, 0.5, 0.75, 1.0].index(level)
                chosen_flat_idx = edge_idx * self.num_levels + level_idx

            # --- 2. CRITIC LOSS: TARGET VALUE ESTIMATION V(s') ---
            with torch.no_grad():
                if done:
                    v_next = torch.tensor(0.0, device=self.device)
                else:
                    next_mask_dict = next_state.get("allowed_destinations", {}).get("antagonist", {})
                    next_allowed = next_mask_dict.get("allowed_edges")
                    if next_allowed is None:
                        next_allowed = list(next_mask_dict.get("levels_by_edge", {}).keys())

                    next_orig = next_mask_dict.get("original_edges")
                    if next_orig is None:
                        next_orig = list(next_state.get("edges", {}).keys())
                    next_budget = trans.info.get("next_antagonist_budget_remaining", remaining_budget)

                    next_num_allowed = len(next_allowed)
                    next_pyg_data = featurize_state(next_state).to(self.device)
                    next_node_ids = list(next_state["nodes"].keys())
                    next_node_to_idx = {nid: idx for idx, nid in enumerate(next_node_ids)}

                    if next_num_allowed == 0:
                        # Only wait action is allowed
                        q1_target_wait, _ = self.target_q1(
                            next_pyg_data, next_orig, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        q2_target_wait, _ = self.target_q2(
                            next_pyg_data, next_orig, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        min_q_wait = torch.min(q1_target_wait[0], q2_target_wait[0])
                        v_next = min_q_wait  # log pi(wait|s) = log(1.0) = 0
                    else:
                        next_edge_probs, next_level_probs, _ = self.actor(
                            next_pyg_data, next_orig, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        next_q1_edge, next_q1_level = self.target_q1(
                            next_pyg_data, next_orig, next_node_to_idx, next_allowed, next_budget, self.level_costs
                        )
                        next_q2_edge, next_q2_level = self.target_q2(
                            next_pyg_data, next_orig, next_node_to_idx, next_allowed, next_budget, self.level_costs
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
            q1_edge, q1_level = self.q1(
                pyg_data, original_edges, node_to_idx, allowed_edges, remaining_budget, self.level_costs
            )
            q2_edge, q2_level = self.q2(
                pyg_data, original_edges, node_to_idx, allowed_edges, remaining_budget, self.level_costs
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
            edge_probs, level_probs, _ = self.actor(
                pyg_data, original_edges, node_to_idx, allowed_edges, remaining_budget, self.level_costs
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
                        # Dynamic target entropy over flat options size
                        target_entropy = -0.98 * torch.log(torch.tensor(1.0 / len(flat_probs), device=self.device))
                    alpha_loss = -self.log_alpha * (entropy - target_entropy).detach()
                    alpha_losses.append(alpha_loss)

        if not critic_losses:
            return None

        # --- 5. BACKPROPAGATION ---
        total_critic_loss = torch.stack(critic_losses).mean()
        self.critic_optimizer.zero_grad()
        total_critic_loss.backward()
        
        critic_grad_norm = sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q1.parameters() if p.grad is not None)
        critic_grad_norm += sum(p.grad.detach().data.norm(2).item() ** 2 for p in self.q2.parameters() if p.grad is not None)
        critic_grad_norm = critic_grad_norm ** 0.5
        
        self.critic_optimizer.step()

        if actor_losses:
            total_actor_loss = torch.stack(actor_losses).mean()
            self.actor_optimizer.zero_grad()
            total_actor_loss.backward()
            
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
