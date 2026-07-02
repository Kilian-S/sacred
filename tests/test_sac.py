"""Unit tests for ProtagonistSAC and AntagonistSAC implementations."""

import unittest
import torch

from src.env.toy_graph import make_toy_graph_env
from src.env.smdp_wrapper import SMDPTransition
from src.agents.sac import ProtagonistSAC, AntagonistSAC, ReplayBuffer


class TestSAC(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_toy_graph_env(num_trucks=2)
        self.obs = self.env.observe()
        self.node_ids = list(self.obs["nodes"].keys())
        self.original_edges = list(self.obs["edges"].keys())

    def test_replay_buffer(self) -> None:
        buffer = ReplayBuffer(capacity=10)
        self.assertEqual(len(buffer), 0)

        # Mock transition
        transition = SMDPTransition(
            agent="protagonist",
            state=self.obs,
            action={0: "a"},
            reward=1.0,
            next_state=self.obs,
            done=False,
            elapsed_ticks=5,
            action_mask={"protagonist": {0: ["a", "b"]}},
            info={}
        )

        buffer.push(transition)
        self.assertEqual(len(buffer), 1)

        batch = buffer.sample(1)
        self.assertEqual(len(batch), 1)
        self.assertEqual(batch[0].agent, "protagonist")

    def test_protagonist_sac_action_selection(self) -> None:
        agent = ProtagonistSAC(
            node_in_dim=11,
            edge_in_dim=2,
            hidden_dim=16,
            num_layers=1,
            heads=1,
            autotune_alpha=True
        )

        action_mask = {0: ["a", "d", "hub"]}
        obs_with_active = dict(self.obs)
        obs_with_active["active_truck"] = 0

        # Test stochastic action selection
        action = agent.select_action(obs_with_active, action_mask, deterministic=False)
        self.assertIn(0, action)
        self.assertIn(action[0], ["a", "d", "hub"])

        # Test deterministic action selection
        action_det = agent.select_action(obs_with_active, action_mask, deterministic=True)
        self.assertIn(0, action_det)
        self.assertIn(action_det[0], ["a", "d", "hub"])

    def test_protagonist_sac_update(self) -> None:
        agent = ProtagonistSAC(
            node_in_dim=11,
            edge_in_dim=2,
            hidden_dim=16,
            num_layers=1,
            heads=1,
            autotune_alpha=True
        )

        obs_state = dict(self.obs)
        obs_state["active_truck"] = 0

        next_obs_state = dict(self.obs)
        next_obs_state["active_truck"] = 1

        # We construct transitions for replay buffer
        t1 = SMDPTransition(
            agent="protagonist",
            state=obs_state,
            action={0: "a"},
            reward=1.5,
            next_state=next_obs_state,
            done=False,
            elapsed_ticks=10,
            action_mask={"protagonist": {0: ["a", "d", "hub"], 1: ["b", "c"]}},
            info={}
        )

        t2 = SMDPTransition(
            agent="protagonist",
            state=obs_state,
            action={0: "d"},
            reward=-0.5,
            next_state=next_obs_state,
            done=True,  # Test terminal state discounting
            elapsed_ticks=2,
            action_mask={"protagonist": {0: ["a", "d", "hub"], 1: ["b", "c"]}},
            info={}
        )

        # Push to buffer
        agent.replay_buffer.push(t1)
        agent.replay_buffer.push(t2)

        # Update and check loss reporting
        metrics = agent.update(batch_size=2)
        self.assertIsNotNone(metrics)
        self.assertIn("protag_critic_loss", metrics)
        self.assertIn("protag_actor_loss", metrics)
        self.assertIn("protag_alpha_loss", metrics)
        self.assertIn("protag_alpha", metrics)

        # Losses should be valid numbers
        self.assertFalse(torch.isnan(torch.tensor(metrics["protag_critic_loss"])))
        self.assertFalse(torch.isnan(torch.tensor(metrics["protag_actor_loss"])))

        agent = AntagonistSAC(
            node_in_dim=11,
            edge_in_dim=2,
            hidden_dim=16,
            num_layers=1,
            heads=1,
            num_congestion_levels=4,
            autotune_alpha=True
        )

        action_mask = {
            "allowed_edges": self.original_edges[:3],
            "original_edges": self.original_edges
        }

        # Test action selection (stochastic)
        action = agent.select_action(self.obs, action_mask, remaining_budget=50.0, deterministic=False)
        if action is not None:
            edge, level = action
            self.assertIn(edge, self.original_edges[:3])
            self.assertIn(level, [0.25, 0.5, 0.75, 1.0])

        # Test budget limits masking: budget = 0.05, so only wait (None) or cheapest congestion option (level 0.25 cost = 0.25*12*0.015 = 0.045) should be valid
        # Let's set extremely low budget (e.g. 0.01) so all level costs exceed remaining budget
        action_low = agent.select_action(self.obs, action_mask, remaining_budget=0.01, deterministic=False)
        self.assertIsNone(action_low)  # Must fall back to "wait" (None) because of budget constraint

    def test_antagonist_sac_update(self) -> None:
        agent = AntagonistSAC(
            node_in_dim=11,
            edge_in_dim=2,
            hidden_dim=16,
            num_layers=1,
            heads=1,
            num_congestion_levels=4,
            autotune_alpha=True
        )

        allowed_edges = self.original_edges[:3]
        action_mask = {
            "antagonist": {
                "allowed_edges": allowed_edges,
                "original_edges": self.original_edges
            }
        }

        t1 = SMDPTransition(
            agent="antagonist",
            state=self.obs,
            action=(self.original_edges[0], 0.5),
            reward=-1.2,
            next_state=self.obs,
            done=False,
            elapsed_ticks=12,
            action_mask=action_mask,
            info={"antagonist_budget_remaining": 30.0, "next_antagonist_budget_remaining": 20.0}
        )

        t2 = SMDPTransition(
            agent="antagonist",
            state=self.obs,
            action=None,  # "wait" action
            reward=0.0,
            next_state=self.obs,
            done=True,
            elapsed_ticks=1,
            action_mask=action_mask,
            info={"antagonist_budget_remaining": 20.0, "next_antagonist_budget_remaining": 20.0}
        )

        agent.replay_buffer.push(t1)
        agent.replay_buffer.push(t2)

        metrics = agent.update(batch_size=2)
        self.assertIsNotNone(metrics)
        self.assertIn("antag_critic_loss", metrics)
        self.assertIn("antag_actor_loss", metrics)
        self.assertIn("antag_alpha_loss", metrics)
        self.assertIn("antag_alpha", metrics)

        self.assertFalse(torch.isnan(torch.tensor(metrics["antag_critic_loss"])))
        self.assertFalse(torch.isnan(torch.tensor(metrics["antag_actor_loss"])))


if __name__ == "__main__":
    unittest.main()
