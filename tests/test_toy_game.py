"""Tests for the deterministic SACRED toy game."""

import unittest

from src.env.multi_agent import (
    GameConfig,
    NearestDemandProtagonist,
    NoOpAntagonist,
    RouteInterceptingAntagonist,
    SacredToyGame,
    describe_game_tick,
)
from src.env.toy_graph import make_toy_graph_env


class ToyGraphTests(unittest.TestCase):
    def test_fixture_has_depot_demands_and_positive_edges(self) -> None:
        env = make_toy_graph_env()
        depot_nodes = [node for node, data in env.graph.nodes(data=True) if data["has_depot"]]
        demand_nodes = [node for node, data in env.graph.nodes(data=True) if data["demand"] > 0]

        self.assertEqual(depot_nodes, ["depot"])
        self.assertGreaterEqual(env.graph.number_of_nodes(), 8)
        self.assertGreaterEqual(len(demand_nodes), 5)
        for _, _, data in env.graph.edges(data=True):
            self.assertGreater(data["distance"], 0)


class RuleAgentTests(unittest.TestCase):
    def test_protagonist_dispatches_idle_trucks_to_demand_nodes(self) -> None:
        env = make_toy_graph_env(num_trucks=2)
        actions = NearestDemandProtagonist().act(env)

        self.assertEqual(set(actions), {0, 1})
        for destination in actions.values():
            self.assertGreater(env.graph.nodes[destination]["demand"], 0)

    def test_protagonist_sends_empty_truck_to_depot(self) -> None:
        env = make_toy_graph_env(num_trucks=1)
        env.trucks[0].current_node = "b"
        env.trucks[0].load = 0.0

        actions = NearestDemandProtagonist().act(env)

        self.assertEqual(actions, {0: "depot"})

    def test_antagonist_respects_budget(self) -> None:
        game = SacredToyGame(config=GameConfig(congestion_budget=0.0))
        game.reset()

        action = RouteInterceptingAntagonist().act(game.env, game)

        self.assertEqual(action, {})


class SacredToyGameTests(unittest.TestCase):
    def test_clean_episode_serves_all_demand(self) -> None:
        game = SacredToyGame(
            antagonist=NoOpAntagonist(),
            config=GameConfig(max_ticks=240),
        )

        metrics = game.run_episode()

        self.assertEqual(metrics.done_reason, "served_all_demand")
        self.assertEqual(metrics.total_delivery, 8.0)
        self.assertEqual(metrics.congestion_events, 0)
        self.assertTrue(
            all(
                truck.current_node == game.env.depot_node and truck.edge is None
                for truck in game.env.trucks.values()
            )
        )

    def test_adversarial_episode_uses_budget_and_produces_metrics(self) -> None:
        game = SacredToyGame(config=GameConfig(max_ticks=80, congestion_budget=20.0))

        metrics = game.run_episode()

        self.assertGreater(metrics.ticks, 0)
        self.assertGreater(metrics.total_delivery, 0)
        self.assertGreater(metrics.congestion_events, 0)
        self.assertGreater(metrics.congestion_budget_used, 0.0)

    def test_tick_description_reports_agent_activity(self) -> None:
        game = SacredToyGame(config=GameConfig(max_ticks=20))
        game.reset()

        tick = game.step()
        messages = describe_game_tick(tick)

        self.assertTrue(any(role == "P" and "fulfilling customer" in message for role, message in messages))
        self.assertTrue(any(role == "A" and "blocking edge" in message for role, message in messages))

    def test_tick_description_reports_return_and_reload(self) -> None:
        game = SacredToyGame(antagonist=NoOpAntagonist(), config=GameConfig(max_ticks=60))
        game.reset()
        return_messages = []
        reload_messages = []

        for _ in range(60):
            tick = game.step()
            messages = describe_game_tick(tick)
            return_messages.extend(message for role, message in messages if role == "P" and "returning to depot" in message)
            reload_messages.extend(message for role, message in messages if role == "P" and "reloaded" in message)
            if return_messages and reload_messages:
                break

        self.assertTrue(return_messages)
        self.assertTrue(reload_messages)


if __name__ == "__main__":
    unittest.main()
