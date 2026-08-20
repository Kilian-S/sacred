"""Tests for the Stage-0 validation rung: the latency reward and the env factory.

Stage 0 is the single-truck validation rung. These tests pin that the latency reward mode
telescopes to total delivery latency and that the cluster factory builds the intended geometry,
while confirming that the legacy static-problem reward path is untouched by default.
"""

import math
import unittest

from src.env.graph_env import GraphEnv
from src.env.smdp_wrapper import DecisionType, SMDPConfig, SMDPDecisionWrapper
from src.baselines.greedy_dispatch import (
    greedy_next_hop_policy,
    greedy_protagonist_policy,
    no_antagonist_policy,
    run_episode,
)
from src.envs.stage0_factory import make_stage0_env, make_stage0_nexthop_env


def _tiny_shuttle_env() -> GraphEnv:
    """Depot + two unit requests at different distances; capacity-1 shuttle."""
    return GraphEnv(
        nodes={
            "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
            "near": {"x": 4.0, "y": 0.0, "demand": 1.0, "has_depot": False},
            "far": {"x": 10.0, "y": 0.0, "demand": 1.0, "has_depot": False},
        },
        edges=[
            ("depot", "near", {"distance": 4.0}),
            ("near", "far", {"distance": 6.0}),
        ],
        num_trucks=1,
        truck_capacity=1.0,
        truck_speed=1.0,
        max_time=200,
    )


def _latency_config(max_ticks: int = 200) -> SMDPConfig:
    return SMDPConfig(
        max_ticks=max_ticks,
        reward_mode="latency",
        antagonist_interval=20,
        congestion_duration=30,
        congestion_budget=300.0,
        congestion_cooldown=0,
        congestion_cost=0.1,
        congestion_levels=(0.25, 0.5, 0.75, 1.0),
    )


class LatencyRewardTest(unittest.TestCase):
    def test_default_reward_mode_is_legacy(self) -> None:
        # The static-problem baseline must be untouched unless latency is opted into.
        self.assertEqual(SMDPConfig().reward_mode, "legacy")

    def test_latency_reward_telescopes_to_total_latency(self) -> None:
        # Drive greedy to full completion, then check the exact telescoping identity:
        # sum over ticks of outstanding-count == sum over requests of (delivery_tick - 1),
        # i.e. total_wait == sum(delivery_ticks) - num_requests  (all arrivals at t=0).
        smdp = SMDPDecisionWrapper(env_factory=_tiny_shuttle_env, config=_latency_config())
        num_requests = smdp._outstanding_requests()

        event = smdp.reset_decision_env()
        delivery_ticks: list[int] = []

        def scan(ev) -> None:
            for tick_info in ev.info.get("events", []):
                for _ in tick_info.get("deliveries", []):
                    delivery_ticks.append(tick_info["time"])

        scan(event)
        ep_reward = 0.0
        policy = greedy_protagonist_policy(smdp)
        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                ne, tr = smdp.step_protagonist(policy(event))
                ep_reward += tr.reward
            elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
                ne, _ = smdp.step_antagonist(None)
                ep_reward += ne.protagonist_reward
            else:
                ne = smdp.advance_until_decision()
            scan(ne)
            event = ne

        total_wait = -ep_reward
        self.assertEqual(len(delivery_ticks), num_requests, "not all requests delivered")
        self.assertAlmostEqual(total_wait, sum(delivery_ticks) - num_requests, places=6)
        # Latency reward is non-positive every tick, so total_wait is strictly positive here.
        self.assertGreater(total_wait, 0.0)

    def test_latency_reward_zero_without_demand(self) -> None:
        # With no outstanding requests, the per-tick latency reward is exactly zero.
        # A depot plus a single non-demand node so the graph has an edge to tick along.
        def idle_env() -> GraphEnv:
            return GraphEnv(
                nodes={
                    "depot": {"x": 0.0, "y": 0.0, "demand": 0.0, "has_depot": True},
                    "empty": {"x": 1.0, "y": 0.0, "demand": 0.0, "has_depot": False},
                },
                edges=[("depot", "empty", {"distance": 1.0})],
                num_trucks=1,
                truck_capacity=1.0,
                max_time=50,
            )

        smdp = SMDPDecisionWrapper(env_factory=idle_env, config=_latency_config(max_ticks=50))
        res = run_episode(smdp, greedy_protagonist_policy(smdp), no_antagonist_policy)
        self.assertEqual(res["total_wait"], 0.0)
        self.assertEqual(res["num_requests"], 0)

    def test_legacy_reward_formula_unchanged(self) -> None:
        # One legacy step must equal delivery_reward*delivered - time_penalty - penalty*remaining.
        cfg = SMDPConfig(max_ticks=50, delivery_reward=10.0, time_penalty=1.0, remaining_demand_penalty=0.5)
        smdp = SMDPDecisionWrapper(env_factory=_tiny_shuttle_env, config=cfg)
        smdp.reset_decision_env()
        step = smdp.env.step()  # one tick, no dispatch, nothing delivered
        smdp._accumulate_step(step, antagonist_action={})
        remaining = smdp._remaining_demand()  # 2 unit requests still outstanding
        expected = (10.0 * 0.0) - 1.0 - (0.5 * remaining)
        self.assertAlmostEqual(smdp._accumulated_protagonist_reward, expected, places=6)


class Stage0FactoryTest(unittest.TestCase):
    def test_factory_geometry(self) -> None:
        env = make_stage0_env(cluster_size=8)
        # One truck, one depot, capacity-1 shuttle.
        self.assertEqual(env.num_trucks, 1)
        self.assertEqual(env.truck_capacity, 1.0)
        self.assertEqual(env.depot_node, env.stage0_depot)
        # Deterministic hotspot = densest node on the Kaliningrad heatmap.
        self.assertEqual(env.stage0_hotspot, "284")
        # Exactly cluster_size unit requests; depot carries none.
        self.assertEqual(len(env.stage0_cluster), 8)
        self.assertNotIn(env.stage0_depot, env.stage0_cluster)
        self.assertAlmostEqual(env.remaining_demand, 8.0, places=6)
        for node_id in env.stage0_cluster:
            self.assertAlmostEqual(env.graph.nodes[node_id]["demand"], 1.0, places=6)

    def test_factory_is_deterministic(self) -> None:
        a = make_stage0_env()
        b = make_stage0_env()
        self.assertEqual(a.stage0_cluster, b.stage0_cluster)
        self.assertEqual(a.stage0_hotspot, b.stage0_hotspot)


class NextHopTest(unittest.TestCase):
    def _nexthop_cfg(self, max_ticks: int = 600) -> SMDPConfig:
        return SMDPConfig(max_ticks=max_ticks, reward_mode="latency", routing_mode="next_hop",
                          antagonist_interval=20, congestion_duration=30, congestion_budget=300.0,
                          congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(0.25, 0.5, 0.75, 1.0))

    def test_dispatch_truck_edge_forces_direct_edge(self) -> None:
        # The anti-A* invariant: next-hop must commit to the chosen direct edge even when a
        # detour is strictly shorter (otherwise the policy never learns to route around).
        env = GraphEnv(
            nodes={"A": {"x": 0, "y": 0, "demand": 0, "has_depot": True},
                   "B": {"x": 2, "y": 0, "demand": 1, "has_depot": False},
                   "M": {"x": 1, "y": 1, "demand": 0, "has_depot": False}},
            edges=[("A", "B", {"distance": 10.0}), ("A", "M", {"distance": 1.0}), ("M", "B", {"distance": 1.0})],
            num_trucks=1, truck_capacity=1.0, truck_starting_nodes=["A"], truck_speed=100.0, max_time=50)
        env.dispatch_truck_edge(0, "B")  # A->B direct is 10; A->M->B is 2. Must still take A-B.
        self.assertEqual(env.trucks[0].edge, ("A", "B"))
        with self.assertRaises(ValueError):
            env2 = make_stage0_nexthop_env()
            env2.dispatch_truck_edge(0, env2.stage0_target)  # target is not adjacent to depot

    def test_next_hop_mask_is_neighbors(self) -> None:
        smdp = SMDPDecisionWrapper(env_factory=make_stage0_nexthop_env, config=self._nexthop_cfg())
        event = smdp.reset_decision_env()
        mask = event.protagonist_action_mask
        depot = smdp.env.stage0_depot
        expected = sorted(smdp.env.graph.neighbors(depot), key=repr)
        self.assertEqual(mask[0], expected)
        # The 14->82 corridor: depot has exactly the two route entrances.
        self.assertEqual(set(mask[0]), {"11", "15"})

    def test_next_hop_episode_delivers_and_telescopes(self) -> None:
        smdp = SMDPDecisionWrapper(env_factory=make_stage0_nexthop_env, config=self._nexthop_cfg())
        num_units = int(round(smdp.env_factory().remaining_demand))
        event = smdp.reset_decision_env()
        delivery_ticks: list[int] = []

        def scan(ev) -> None:
            for ti in ev.info.get("events", []):
                for _ in ti.get("deliveries", []):
                    delivery_ticks.append(ti["time"])

        scan(event)
        ep_reward = 0.0
        policy = greedy_next_hop_policy(smdp)
        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                ne, tr = smdp.step_protagonist(policy(event))
                ep_reward += tr.reward
            elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
                ne, _ = smdp.step_antagonist(None)
                ep_reward += ne.protagonist_reward
            else:
                ne = smdp.advance_until_decision()
            scan(ne)
            event = ne

        self.assertEqual(len(delivery_ticks), num_units)  # all units delivered (capacity-1 trips)
        self.assertAlmostEqual(-ep_reward, sum(delivery_ticks) - num_units, places=6)

    def test_nexthop_factory_focused_two_route(self) -> None:
        env = make_stage0_nexthop_env()
        self.assertEqual(env.num_trucks, 1)
        self.assertEqual(env.depot_node, "14")
        self.assertEqual(env.stage0_target, "82")
        # All demand focused on the single target node.
        self.assertAlmostEqual(env.remaining_demand, 12.0, places=6)
        self.assertAlmostEqual(env.graph.nodes["82"]["demand"], 12.0, places=6)
        # Depot and target joined by two node-disjoint routes (only endpoints shared).
        import networkx as nx
        p1 = nx.shortest_path(env.graph, "14", "82", weight="distance")
        h = env.graph.copy()
        for i in range(len(p1) - 1):
            h.remove_edge(p1[i], p1[i + 1])
        p2 = nx.shortest_path(h, "14", "82", weight="distance")  # raises if no 2nd route
        self.assertEqual(set(p1) & set(p2), {"14", "82"})


if __name__ == "__main__":
    unittest.main()
