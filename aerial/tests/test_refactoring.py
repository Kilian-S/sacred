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

        self.assertIn(0, mask)
        self.assertIn(1, mask)

        # The toy graph has six customer nodes with positive demand; the depot is excluded
        # because the trucks start there fully loaded.
        allowed = mask[0]
        self.assertEqual(len(allowed), 6)
        self.assertNotIn("depot", allowed)
        self.assertIn("a", allowed)
        self.assertIn("b", allowed)
        self.assertNotIn("hub", allowed)     # zero demand
        self.assertNotIn("bridge", allowed)  # zero demand

    def test_sequential_dispatch_and_state_projection(self) -> None:
        event = self.smdp.reset_decision_env()
        mask = self.smdp.protagonist_action_mask()

        obs_copy = copy.deepcopy(event.observation)
        obs_copy["active_truck"] = 0
        obs_copy["allowed_destinations"] = {"protagonist": dict(mask)}

        pyg_data_0 = featurize_state(obs_copy, active_truck_id=0)
        node_ids = sorted(list(event.observation["nodes"].keys()))
        idx_a = node_ids.index("a")

        # col 7 = is_targeted_by_other, col 8 = unassigned_demand
        self.assertEqual(pyg_data_0.x[idx_a, 7].item(), 0.0)
        self.assertEqual(pyg_data_0.x[idx_a, 8].item(), 1.0)

        # Project truck 0's commitment to 'a', then featurise for truck 1.
        obs_copy["trucks"][0]["destination"] = "a"
        obs_copy["trucks"][0]["current_node"] = None

        pyg_data_1 = featurize_state(obs_copy, active_truck_id=1)

        self.assertEqual(pyg_data_1.x[idx_a, 7].item(), 1.0)
        self.assertEqual(pyg_data_1.x[idx_a, 8].item(), 0.0)  # demand 1.0 - capacity 1.0

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

        dispatch_actions = {0: "b", 1: "e"}
        next_event, transition = smdp.step_protagonist(dispatch_actions)

        # At tick 5 both trucks are still short of their destinations (reached at tick 22).
        truck0 = smdp.env.trucks[0]
        truck1 = smdp.env.trucks[1]
        self.assertFalse(truck0.is_idle)
        self.assertFalse(truck1.is_idle)
        self.assertEqual(truck0.destination, "b")

        # Edge ('a', 'b') lies on truck 0's Dijkstra path depot->a->b, ahead of its current edge.
        congestion_edge = smdp.env._edge_key("a", "b")

        # The reroute happens inside this step (a no-movement sequential epoch), so assert
        # immediately, before time advances and the truck actually reaches 'a'.
        antag_action = (congestion_edge, 1.0)
        smdp.step_antagonist(antag_action)

        # Truck 0 now heads for the junction 'a' before the congested edge, not 'b'.
        self.assertEqual(truck0.destination, "a")
        self.assertFalse(truck0.is_idle)
        self.assertIsNone(truck0.current_node)
        self.assertEqual(tuple(truck0.path), ("depot", "a"))  # path truncated at the reroute point

    def test_unassigned_demand_masking(self) -> None:
        event = self.smdp.reset_decision_env()

        mask_init = self.smdp.protagonist_action_mask()
        self.assertIn("a", mask_init[0])
        self.assertIn("a", mask_init[1])

        # Commit truck 0 to 'a', whose demand of 1.0 its load of 1.0 fully covers.
        self.smdp.env.trucks[0].destination = "a"
        self.smdp.env.trucks[0].current_node = None

        mask_new = self.smdp.protagonist_action_mask()

        self.assertNotIn("a", mask_new[1])   # no unassigned demand left at 'a'
        self.assertIn("b", mask_new[1])      # 'b' has demand 2.0, still available


if __name__ == "__main__":
    unittest.main()
