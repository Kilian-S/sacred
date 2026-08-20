"""Assignment probe: the two-depot factory, greedy insertion claiming sequentially so it never
double-assigns, and ERB demos from the shared transition builder staying compatible with a SAC
update."""

import unittest

from src.env.smdp_wrapper import DecisionType, SMDPConfig, SMDPDecisionWrapper
from src.envs.assignment_factory import make_assignment_env
from src.agents.transition_builder import collect_protagonist_transitions
from src.baselines.greedy_dispatch import (
    _congestion_aware_distance, _id_key, greedy_insertion_policy, no_antagonist_policy, run_episode,
)


def _cfg() -> SMDPConfig:
    return SMDPConfig(max_ticks=800, reward_mode="latency", routing_mode="destination",
                      antagonist_interval=20, congestion_duration=30, congestion_budget=400.0,
                      congestion_cooldown=0, congestion_cost=0.1, congestion_levels=(0.25, 0.5, 0.75, 1.0))


class AssignmentFactoryTest(unittest.TestCase):
    def test_two_depots_two_trucks_contested_demand(self) -> None:
        env = make_assignment_env()
        self.assertEqual(env.num_trucks, 2)
        self.assertEqual(set(env.assignment_depots), {"110", "135"})
        self.assertAlmostEqual(env.remaining_demand, float(len(env.assignment_demand)), places=6)
        for n in env.assignment_demand:
            self.assertGreater(env.graph.nodes[n]["demand"], 0.0)


class GreedyInsertionTest(unittest.TestCase):
    def test_delivers_all_no_double_assign(self) -> None:
        smdp = SMDPDecisionWrapper(env_factory=make_assignment_env, config=_cfg())
        r = run_episode(smdp, greedy_insertion_policy(smdp), no_antagonist_policy)
        self.assertEqual(r["delivered"], r["num_requests"])  # all served

    def test_sequential_claiming_no_two_trucks_same_request(self) -> None:
        smdp = SMDPDecisionWrapper(env_factory=make_assignment_env, config=_cfg())
        policy = greedy_insertion_policy(smdp)
        event = smdp.reset_decision_env()
        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                a = policy(event)
                demand_targets = [n for n in a.values() if smdp.env.graph.nodes[n]["demand"] > 0.0]
                self.assertEqual(len(demand_targets), len(set(demand_targets)), "double-assignment!")
                event, _ = smdp.step_protagonist(a)
            elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
                event, _ = smdp.step_antagonist(None)
            else:
                event = smdp.advance_until_decision()


class ErbDemoFormatTest(unittest.TestCase):
    def _greedy_choose(self, smdp):
        def choose(projected_obs, truck_mask, truck_id):
            env = smdp.env
            dests = truck_mask.get(truck_id, [])
            if not dests:
                return {}
            reqs = [d for d in dests if env.graph.nodes[d]["demand"] > 0.0]
            src = env.trucks[truck_id].current_node
            if reqs and src is not None:
                return {truck_id: min(reqs, key=lambda d: (_congestion_aware_distance(env, src, d), _id_key(d)))}
            return {truck_id: dests[0]}
        return choose

    def test_demos_have_required_fields_and_no_double_claim(self) -> None:
        smdp = SMDPDecisionWrapper(env_factory=make_assignment_env, config=_cfg())
        choose = self._greedy_choose(smdp)
        event = smdp.reset_decision_env()
        demos = []
        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                event, ts = collect_protagonist_transitions(smdp, event, choose)
                demos.extend(ts)
            elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
                event, _ = smdp.step_antagonist(None)
            else:
                event = smdp.advance_until_decision()
        self.assertGreater(len(demos), 0)
        for t in demos:
            self.assertEqual(t.agent, "protagonist")
            at = t.state.get("active_truck")
            self.assertIsNotNone(at)
            # SAC.update() indexes this field, so the chosen node must be in the stored mask
            allowed = t.action_mask["protagonist"][at]
            self.assertIn(t.action[at], allowed)

    def test_demos_are_sac_update_compatible(self) -> None:
        from src.agents.sac import ProtagonistSAC
        smdp = SMDPDecisionWrapper(env_factory=make_assignment_env, config=_cfg())
        choose = self._greedy_choose(smdp)
        agent = ProtagonistSAC(node_in_dim=13, edge_in_dim=4, hidden_dim=32, num_layers=2, heads=4, device="cpu")
        event = smdp.reset_decision_env()
        n = 0
        while not event.done and n < 40:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                event, ts = collect_protagonist_transitions(smdp, event, choose)
                for t in ts:
                    agent.replay_buffer.push(t); n += 1
            elif event.decision_type == DecisionType.ANTAGONIST_DECISION:
                event, _ = smdp.step_antagonist(None)
            else:
                event = smdp.advance_until_decision()
        metrics = agent.update(8)  # must run without shape/index error on demo transitions
        self.assertIsNotNone(metrics)
        self.assertTrue(all(v == v for v in metrics.values()))  # no NaN


if __name__ == "__main__":
    unittest.main()
