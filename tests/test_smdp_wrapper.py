"""Tests for the SMDP decision wrapper."""

import unittest

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import DecisionType, SMDPConfig, SMDPDecisionWrapper


def direct_env(distance: float = 10.0, speed: float = 1.0) -> GraphEnv:
    return GraphEnv(
        nodes={
            "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
            "customer": {"x": distance, "y": 0.0, "demand": 1.0, "has_depot": False},
        },
        edges=[("depot", "customer", {"distance": distance})],
        truck_speed=speed,
        truck_capacity=1.0,
        max_time=240,
    )


def intermediate_env() -> GraphEnv:
    return GraphEnv(
        nodes={
            "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
            "mid": {"x": 5.0, "y": 0.0, "demand": 0.0, "has_depot": False},
            "customer": {"x": 10.0, "y": 0.0, "demand": 1.0, "has_depot": False},
        },
        edges=[
            ("depot", "mid", {"distance": 5.0}),
            ("mid", "customer", {"distance": 5.0}),
        ],
        truck_speed=1.0,
        truck_capacity=1.0,
        max_time=240,
    )


class SMDPWrapperTests(unittest.TestCase):
    def test_reset_returns_initial_protagonist_decision(self) -> None:
        wrapper = SMDPDecisionWrapper(env_factory=direct_env)

        event = wrapper.reset_decision_env()

        self.assertEqual(event.decision_type, DecisionType.PROTAGONIST_DECISION)
        self.assertEqual(event.waiting_trucks, [0])
        self.assertEqual(event.elapsed_ticks, 0)
        self.assertCountEqual(event.protagonist_action_mask[0], ["customer"])

    def test_protagonist_transition_skips_driving_ticks(self) -> None:
        wrapper = SMDPDecisionWrapper(env_factory=lambda: direct_env(distance=10.0, speed=1.0))
        wrapper.reset_decision_env()

        event, transition = wrapper.step_protagonist({0: "customer"})

        self.assertEqual(event.decision_type, DecisionType.PROTAGONIST_DECISION)
        self.assertEqual(event.observation["time"], 10)
        self.assertEqual(event.waiting_trucks, [0])
        self.assertEqual(event.protagonist_action_mask[0], ["depot"])
        self.assertEqual(transition.elapsed_ticks, 10)
        self.assertEqual(transition.next_state["nodes"]["customer"]["demand"], 0.0)

    def test_truck_does_not_pause_at_intermediate_node(self) -> None:
        wrapper = SMDPDecisionWrapper(env_factory=intermediate_env)
        wrapper.reset_decision_env()

        event, _ = wrapper.step_protagonist({0: "customer"})

        self.assertEqual(event.decision_type, DecisionType.PROTAGONIST_DECISION)
        self.assertEqual(event.observation["time"], 10)
        self.assertEqual(event.observation["trucks"][0]["current_node"], "customer")
        self.assertCountEqual(event.protagonist_action_mask[0], ["depot"])

    def test_antagonist_event_fires_at_sixty_ticks(self) -> None:
        wrapper = SMDPDecisionWrapper(
            env_factory=lambda: direct_env(distance=100.0, speed=1.0),
            config=SMDPConfig(antagonist_interval=60, max_ticks=240),
        )
        wrapper.reset_decision_env()

        event, transition = wrapper.step_protagonist({0: "customer"})

        self.assertEqual(event.decision_type, DecisionType.ANTAGONIST_DECISION)
        self.assertEqual(event.observation["time"], 60)
        self.assertEqual(transition.elapsed_ticks, 60)

    def test_simultaneous_event_orders_antagonist_before_protagonist(self) -> None:
        wrapper = SMDPDecisionWrapper(
            env_factory=lambda: direct_env(distance=60.0, speed=1.0),
            config=SMDPConfig(antagonist_interval=60, max_ticks=240),
        )
        wrapper.reset_decision_env()
        event, _ = wrapper.step_protagonist({0: "customer"})
        edge = next(iter(event.antagonist_action_mask["levels_by_edge"]))

        self.assertEqual(event.decision_type, DecisionType.BOTH_DECISION)

        next_event, transition = wrapper.step_antagonist((edge, 0.5))

        self.assertEqual(next_event.decision_type, DecisionType.PROTAGONIST_DECISION)
        self.assertEqual(next_event.observation["time"], 60)
        self.assertEqual(next_event.elapsed_ticks, 0)
        self.assertEqual(transition.elapsed_ticks, 0)
        self.assertEqual(next_event.observation["edges"][edge]["congestion_level"], 0.5)

    def test_smdp_discount_uses_elapsed_ticks(self) -> None:
        wrapper = SMDPDecisionWrapper()

        self.assertAlmostEqual(wrapper.smdp_discount(0.99, 35), 0.99**35)


if __name__ == "__main__":
    unittest.main()

