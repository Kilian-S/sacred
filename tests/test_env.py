"""Tests for the headless graph environment."""

import unittest

from src.env.graph_env import GraphEnv


class GraphEnvTests(unittest.TestCase):
    def test_truck_serves_customer_after_exact_edge_travel(self) -> None:
        env = GraphEnv(
            nodes={
                "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
                "customer": {"x": 10.0, "y": 0.0, "demand": 1.0, "has_depot": False},
            },
            edges=[("depot", "customer", {"distance": 10.0, "congestion_level": 0.0})],
            truck_speed=5.0,
        )

        first = env.step(dispatch_actions={0: "customer"})
        self.assertEqual(first.observation["trucks"][0]["edge"], ("depot", "customer"))
        self.assertEqual(first.observation["trucks"][0]["edge_progress"], 5.0)
        self.assertEqual(first.observation["nodes"]["customer"]["demand"], 1.0)

        second = env.step()
        self.assertIsNone(second.observation["trucks"][0]["edge"])
        self.assertEqual(second.observation["trucks"][0]["current_node"], "customer")
        self.assertEqual(second.observation["nodes"]["customer"]["demand"], 0.0)
        self.assertEqual(second.observation["trucks"][0]["load"], 0.0)
        self.assertFalse(second.done)

        third = env.step(dispatch_actions={0: "depot"})
        self.assertEqual(third.observation["trucks"][0]["edge_progress"], 5.0)

        fourth = env.step()
        self.assertEqual(fourth.observation["trucks"][0]["current_node"], "depot")
        self.assertEqual(fourth.observation["trucks"][0]["load"], 1.0)
        self.assertTrue(fourth.done)

    def test_congestion_reduces_distance_per_tick(self) -> None:
        env = GraphEnv(
            nodes={
                "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
                "customer": {"x": 10.0, "y": 0.0, "demand": 1.0, "has_depot": False},
            },
            edges=[("depot", "customer", {"distance": 10.0})],
            truck_speed=10.0,
        )

        result = env.step(
            dispatch_actions={0: "customer"},
            congestion_actions={("depot", "customer"): 0.5},
        )

        self.assertEqual(result.observation["trucks"][0]["edge_progress"], 5.0)
        self.assertEqual(result.info["distance_travelled"], 5.0)

    def test_reset_restores_demands_and_congestion(self) -> None:
        env = GraphEnv(
            nodes={
                "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
                "customer": {"x": 1.0, "y": 0.0, "demand": 1.0, "has_depot": False},
            },
            edges=[("depot", "customer", {"distance": 1.0})],
            truck_speed=1.0,
        )

        env.step(
            dispatch_actions={0: "customer"},
            congestion_actions={("depot", "customer"): 0.25},
        )
        env.set_congestion(("depot", "customer"), 0.75)
        observation = env.reset()

        self.assertEqual(observation["nodes"]["customer"]["demand"], 1.0)
        self.assertEqual(observation["edges"][("customer", "depot")]["congestion_level"], 0.0)
        self.assertEqual(observation["trucks"][0]["load"], 1.0)

    def test_delivery_limited_by_truck_load(self) -> None:
        env = GraphEnv(
            nodes={
                "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
                "customer": {"x": 1.0, "y": 0.0, "demand": 2.0, "has_depot": False},
            },
            edges=[("depot", "customer", {"distance": 1.0})],
            truck_speed=1.0,
            truck_capacity=1.0,
        )

        result = env.step(dispatch_actions={0: "customer"})

        self.assertEqual(result.observation["nodes"]["customer"]["demand"], 1.0)
        self.assertEqual(result.observation["trucks"][0]["load"], 0.0)
        self.assertEqual(result.info["deliveries"][0]["delivered"], 1.0)


if __name__ == "__main__":
    unittest.main()
