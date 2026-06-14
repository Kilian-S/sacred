"""Unit tests for the refactored destination-based protagonist, state projection, and dynamic rerouting."""

import unittest
import copy
import torch

from src.env.toy_graph import make_toy_graph_env
from src.env.smdp_wrapper import SMDPDecisionWrapper, SMDPConfig, DecisionType
from src.agents.networks import featurize_state


class TestProtagonistRefactoring(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SMDPConfig(
            max_ticks=240,
            antagonist_interval=60,
            congestion_budget=72.0,
            delivery_reward=20.0,
            remaining_demand_penalty=0.08,
        )
        self.smdp = SMDPDecisionWrapper(
            env_factory=lambda: make_toy_graph_env(num_trucks=2),
            config=self.config,
        )

    def test_goal_directed_action_mask(self) -> None:
        event = self.smdp.reset_decision_env()
        mask = self.smdp.protagonist_action_mask()

        # Both trucks should be idle initially
        self.assertIn(0, mask)
        self.assertIn(1, mask)

        # Allowed destinations should contain all customer nodes with positive demand.
        # In toy graph, customer nodes with positive demand are: a, b, c, d, e, f.
        # Depot is excluded because the trucks are fully loaded at the depot initially.
        allowed = mask[0]
        self.assertEqual(len(allowed), 6)
        self.assertNotIn("depot", allowed)
        self.assertIn("a", allowed)
        self.assertIn("b", allowed)
        self.assertNotIn("hub", allowed)  # hub has 0.0 demand, should NOT be in mask!
        self.assertNotIn("bridge", allowed)  # bridge has 0.0 demand, should NOT be in mask!

    def test_sequential_dispatch_and_state_projection(self) -> None:
        event = self.smdp.reset_decision_env()
        mask = self.smdp.protagonist_action_mask()

        # Step 1: First truck chooses customer 'a'
        obs_copy = copy.deepcopy(event.observation)
        obs_copy["active_truck"] = 0
        obs_copy["allowed_destinations"] = {"protagonist": dict(mask)}

        # Featurize for truck 0: no commitments yet
        pyg_data_0 = featurize_state(obs_copy, active_truck_id=0)
        node_ids = list(event.observation["nodes"].keys())
        idx_a = node_ids.index("a")
        
        # 'is_targeted_by_other' (column 7) and 'unassigned_demand' (column 8) for node 'a'
        self.assertEqual(pyg_data_0.x[idx_a, 7].item(), 0.0)
        self.assertEqual(pyg_data_0.x[idx_a, 8].item(), 1.0)  # full demand of 1.0

        # Now project commitment of Truck 0 targeting 'a'
        obs_copy["trucks"][0]["destination"] = "a"
        obs_copy["trucks"][0]["current_node"] = None

        # Featurize for truck 1: should see truck 0's commitment
        pyg_data_1 = featurize_state(obs_copy, active_truck_id=1)
        
        # Node 'a' should be targeted by other
        self.assertEqual(pyg_data_1.x[idx_a, 7].item(), 1.0)
        # unassigned demand at 'a' should be 0.0 (demand 1.0 - capacity 1.0)
        self.assertEqual(pyg_data_1.x[idx_a, 8].item(), 0.0)

    def test_gps_style_dynamic_rerouting(self) -> None:
        config = SMDPConfig(
            max_ticks=240,
            antagonist_interval=5,
            congestion_budget=72.0,
            delivery_reward=20.0,
            remaining_demand_penalty=0.08,
        )
        smdp = SMDPDecisionWrapper(
            env_factory=lambda: make_toy_graph_env(num_trucks=2),
            config=config,
        )
        event = smdp.reset_decision_env()
        
        # Dispatch Truck 0 to node 'b' and Truck 1 to node 'e'
        dispatch_actions = {0: "b", 1: "e"}
        next_event, transition = smdp.step_protagonist(dispatch_actions)

        # Confirm trucks are moving (at tick 5, before they reach destination at tick 22)
        truck0 = smdp.env.trucks[0]
        truck1 = smdp.env.trucks[1]
        self.assertFalse(truck0.is_idle)
        self.assertFalse(truck1.is_idle)
        self.assertEqual(truck0.destination, "b")

        # Apply antagonist congestion on edge ('a', 'b') which is on the path to 'b'
        congestion_edge = smdp.env._edge_key("a", "b")
        
        # Set antagonist action
        antag_action = (congestion_edge, 1.0)
        smdp.step_antagonist(antag_action)
        # End sequential decision epoch to advance simulated time
        next_event2, transition2 = smdp.step_antagonist(None)

        # The antagonist step should have intercepted Truck 0 and truncated its path!
        # It should have arrived at the next intersection ('a') and become idle.
        self.assertEqual(truck0.current_node, "a")
        self.assertTrue(truck0.is_idle)
        self.assertIsNone(truck0.destination)

    def test_unassigned_demand_masking(self) -> None:
        event = self.smdp.reset_decision_env()
        
        # Initially, all customer nodes have unassigned demand > 0
        mask_init = self.smdp.protagonist_action_mask()
        self.assertIn("a", mask_init[0])
        self.assertIn("a", mask_init[1])
        
        # Commit Truck 0 to target customer 'a' (which has demand 1.0, Truck 0 carries load 1.0)
        self.smdp.env.trucks[0].destination = "a"
        self.smdp.env.trucks[0].current_node = None
        
        # Now evaluate mask for Truck 1 (which is idle and still deciding)
        mask_new = self.smdp.protagonist_action_mask()
        
        # Customer 'a' should be dynamically excluded from Truck 1's action mask because unassigned demand is 0!
        self.assertNotIn("a", mask_new[1])
        self.assertIn("b", mask_new[1])  # 'b' has demand 2.0, remains available


if __name__ == "__main__":
    unittest.main()
