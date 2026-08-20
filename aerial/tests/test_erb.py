"""Unit tests for ALNS metaheuristic solver and ERB pre-seeding trajectory generator."""

import unittest
import os
import torch
import networkx as nx

from src.env.toy_graph import make_toy_graph_env
from src.env.smdp_wrapper import SMDPDecisionWrapper, SMDPConfig, DecisionType, SMDPTransition
from src.baselines.metaheuristic import AdaptiveLargeNeighborhoodSearchVRP


class TestERB(unittest.TestCase):
    def setUp(self) -> None:
        self.env = make_toy_graph_env(num_trucks=2)

    def test_alns_initialization(self) -> None:
        alns = AdaptiveLargeNeighborhoodSearchVRP(self.env, iterations=10)
        
        self.assertIn(self.env.depot_node, alns.distances)
        self.assertGreater(len(alns.tasks), 0)
        
        self.assertTrue(all(task[0] != self.env.depot_node for task in alns.tasks))
        self.assertTrue(all(task[2] <= 1.0 for task in alns.tasks))

    def test_alns_solve(self) -> None:
        alns = AdaptiveLargeNeighborhoodSearchVRP(self.env, iterations=20)
        best_sol = alns.solve()

        self.assertEqual(len(best_sol), 2)
        self.assertIn(0, best_sol)
        self.assertIn(1, best_sol)

        # every task is assigned to exactly one truck
        total_tasks_assigned = sum(len(seq) for seq in best_sol.values())
        self.assertEqual(total_tasks_assigned, len(alns.tasks))

    def test_physical_route_reconstruction(self) -> None:
        alns = AdaptiveLargeNeighborhoodSearchVRP(self.env, iterations=10)
        best_sol = alns.solve()

        for t_id, task_seq in best_sol.items():
            route = alns.get_physical_route(task_seq)
            
            self.assertEqual(route[0], self.env.depot_node)
            self.assertEqual(route[-1], self.env.depot_node)
            
            for i in range(len(route) - 1):
                u, v = route[i], route[i+1]
                self.assertTrue(self.env.graph.has_edge(u, v))

    def test_erb_transitions_format(self) -> None:
        config = SMDPConfig(max_ticks=60, antagonist_interval=30)
        smdp = SMDPDecisionWrapper(
            env_factory=lambda: make_toy_graph_env(num_trucks=2),
            config=config,
        )

        alns = AdaptiveLargeNeighborhoodSearchVRP(self.env, iterations=10)
        best_sol = alns.solve()
        
        truck_paths = {t_id: alns.get_high_level_destinations(best_sol[t_id]) for t_id in best_sol}
        path_indices = {t_id: 0 for t_id in best_sol}

        event = smdp.reset_decision_env()
        transitions = []

        while not event.done:
            if event.decision_type in (DecisionType.PROTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                import copy
                actions = {}
                projected_obs = copy.deepcopy(event.observation)
                truck_decision_states = {}

                for truck_id in event.waiting_trucks:
                    projected_obs["active_truck"] = truck_id
                    projected_obs["allowed_destinations"] = {"protagonist": dict(event.protagonist_action_mask)}
                    
                    truck_decision_states[truck_id] = copy.deepcopy(projected_obs)

                    path = truck_paths[truck_id]
                    idx = path_indices[truck_id]
                    if idx < len(path):
                        next_node = path[idx]
                        path_indices[truck_id] += 1
                    else:
                        next_node = smdp.env.depot_node
                    
                    actions[truck_id] = next_node
                    
                    # project the commitment into the state stored for this truck
                    projected_obs["trucks"][truck_id]["destination"] = next_node
                    projected_obs["trucks"][truck_id]["current_node"] = None

                next_event, transition = smdp.step_protagonist(actions)
                
                for truck_id in event.waiting_trucks:
                    state_copy = truck_decision_states[truck_id]

                    next_state_copy = dict(next_event.observation)
                    if next_event.waiting_trucks:
                        next_state_copy["active_truck"] = next_event.waiting_trucks[0]
                    else:
                        next_state_copy["active_truck"] = None
                    next_state_copy["allowed_destinations"] = {
                        "protagonist": dict(next_event.protagonist_action_mask)
                    }

                    t_trans = SMDPTransition(
                        agent="protagonist",
                        state=state_copy,
                        action=actions,
                        reward=transition.reward,
                        next_state=next_state_copy,
                        done=transition.done,
                        elapsed_ticks=transition.elapsed_ticks,
                        action_mask={"protagonist": dict(event.protagonist_action_mask)},
                        info=dict(transition.info)
                    )
                    transitions.append(t_trans)
                event = next_event
            elif event.decision_type in (DecisionType.ANTAGONIST_DECISION, DecisionType.BOTH_DECISION):
                next_event, transition = smdp.step_antagonist(None)
                event = next_event
            else:
                event = smdp.advance_until_decision()

        self.assertGreater(len(transitions), 0)
        for t in transitions:
            self.assertEqual(t.agent, "protagonist")
            self.assertIn("active_truck", t.state)
            self.assertIn("allowed_destinations", t.state)
            self.assertIsInstance(t.action, dict)
            self.assertIsInstance(t.reward, float)
            self.assertIsInstance(t.done, bool)
            self.assertIsInstance(t.elapsed_ticks, int)


if __name__ == "__main__":
    unittest.main()
