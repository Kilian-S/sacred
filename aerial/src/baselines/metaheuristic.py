"""ALNS logic for experience replay buffer bootstrapping."""

from __future__ import annotations

import copy
import random
import math
from typing import Any, Mapping
import networkx as nx
import numpy as np

from src.env.graph_env import GraphEnv, NodeId, EdgeId


class AdaptiveLargeNeighborhoodSearchVRP:
    """State-of-the-art Adaptive Large Neighborhood Search (ALNS) solver for the CVRP on graphs.

    Parameters
    ----------
    env:
        The GraphEnv environment instance defining topology and truck speeds.
    iterations:
        Number of ALNS search iterations (default: 300).
    decay:
        Adaptive weight decay factor (default: 0.8).
    annealing_temp:
        Initial temperature for Simulated Annealing acceptance (default: 100.0).
    cooling_rate:
        Cooling rate for Simulated Annealing (default: 0.98).
    """

    def __init__(
        self,
        env: GraphEnv,
        iterations: int = 300,
        decay: float = 0.8,
        annealing_temp: float = 100.0,
        cooling_rate: float = 0.98,
    ) -> None:
        self.env = env
        self.iterations = iterations
        self.decay = decay
        self.temp = annealing_temp
        self.cooling_rate = cooling_rate

        self.depot = env.depot_node
        self.num_trucks = env.num_trucks
        self.capacity = env.truck_capacity

        # 1. Precompute shortest paths and distances using NetworkX
        self.distances = dict(nx.all_pairs_dijkstra_path_length(env.graph, weight="distance"))
        self.paths = dict(nx.all_pairs_dijkstra_path(env.graph, weight="distance"))

        # 2. Featurize tasks (partition demands into 1.0 capacity chunks)
        # Each task is a tuple: (customer_node, task_index_for_node, demand_value)
        self.tasks: list[tuple[NodeId, int, float]] = []
        depot_comp = env.node_to_component[self.depot]
        for node in sorted(list(env.graph.nodes), key=lambda x: str(x)):
            data = env.graph.nodes[node]
            if data.get("has_depot") or data.get("demand", 0) <= 0:
                continue
            if env.node_to_component.get(node, -1) != depot_comp:
                continue
            demand = float(data["demand"])
            # Break demand into chunks of size <= 1.0 (truck capacity)
            num_chunks = math.ceil(demand)
            for i in range(num_chunks):
                chunk_demand = min(1.0, demand - i * 1.0)
                if chunk_demand > 0.001:
                    self.tasks.append((node, i, chunk_demand))

        # 3. Initialize destroy and repair operator weights for Adaptive Engine
        # Destroy operators: 0 = Random, 1 = Worst Cost, 2 = Shaw (Similarity)
        # Repair operators: 0 = Greedy, 1 = Regret-2
        self.destroy_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        self.repair_weights = np.array([1.0, 1.0], dtype=np.float32)
        
        self.destroy_scores = np.zeros(3, dtype=np.float32)
        self.repair_scores = np.zeros(2, dtype=np.float32)
        
        self.destroy_counts = np.zeros(3, dtype=np.int32)
        self.repair_counts = np.zeros(2, dtype=np.int32)

    def solve(self) -> dict[int, list[tuple[NodeId, int, float]]]:
        """Run the ALNS loop and return the best task sequence for each truck."""
        if not self.tasks:
            return {t_id: [] for t_id in range(self.num_trucks)}

        # 1. Generate constructive initial solution
        current_sol = self._nearest_neighbor_initial()
        current_cost = self._evaluate_solution(current_sol)

        best_sol = {k: list(v) for k, v in current_sol.items()}
        best_cost = current_cost

        # Determine number of tasks to destroy (typically 20% to 30% of tasks)
        n_remove = max(1, min(len(self.tasks) // 2, int(len(self.tasks) * 0.25)))

        for r in range(self.iterations):
            # A. Select destroy and repair operators using roulette wheel selection
            d_op = self._roulette_wheel_select(self.destroy_weights)
            r_op = self._roulette_wheel_select(self.repair_weights)

            self.destroy_counts[d_op] += 1
            self.repair_counts[r_op] += 1

            # B. Apply Destroy
            partial_sol, removed = self._apply_destroy(current_sol, d_op, n_remove)

            # C. Apply Repair
            candidate_sol = self._apply_repair(partial_sol, r_op, removed)
            candidate_cost = self._evaluate_solution(candidate_sol)

            # D. Evaluate acceptance (Simulated Annealing)
            score = 0
            if candidate_cost < best_cost - 1e-6:
                # New global best
                best_sol = {k: list(v) for k, v in candidate_sol.items()}
                best_cost = candidate_cost
                current_sol = {k: list(v) for k, v in candidate_sol.items()}
                current_cost = candidate_cost
                score = 3
            elif candidate_cost < current_cost - 1e-6:
                # Improving current solution
                current_sol = {k: list(v) for k, v in candidate_sol.items()}
                current_cost = candidate_cost
                score = 2
            else:
                # Acceptance threshold under Simulated Annealing criteria
                delta = candidate_cost - current_cost
                prob = math.exp(-delta / max(1e-6, self.temp))
                if random.random() < prob:
                    current_sol = {k: list(v) for k, v in candidate_sol.items()}
                    current_cost = candidate_cost
                    score = 1

            # E. Update Operator Scores
            self.destroy_scores[d_op] += score
            self.repair_scores[r_op] += score

            # Cool the temperature
            self.temp *= self.cooling_rate

            # Periodically update adaptive weights (every 50 iterations)
            if (r + 1) % 50 == 0:
                self._update_adaptive_weights()

        return best_sol

    def get_physical_route(self, task_sequence: list[tuple[NodeId, int, float]]) -> list[NodeId]:
        """Translate a sequence of task dispatches into a tick-by-tick node path."""
        route = [self.depot]
        curr_load = self.capacity

        for node, _, demand in task_sequence:
            # Check capacity: if depleted, we must reload at the depot first
            if curr_load < demand - 1e-6:
                # Travel back to depot to reload
                if route[-1] != self.depot:
                    subpath = self.paths[route[-1]][self.depot]
                    route.extend(subpath[1:])
                curr_load = self.capacity

            # Travel to target node
            if route[-1] != node:
                subpath = self.paths[route[-1]][node]
                route.extend(subpath[1:])
            
            # Fulfill demand
            curr_load -= demand

        # Finally, return back to depot to complete the VRP cycle
        if route[-1] != self.depot:
            subpath = self.paths[route[-1]][self.depot]
            route.extend(subpath[1:])

        return route

    def get_high_level_destinations(self, task_sequence: list[tuple[NodeId, int, float]]) -> list[NodeId]:
        """Translate a sequence of VRP tasks into a sequence of high-level destination nodes."""
        destinations = []
        curr_load = self.capacity

        for node, _, demand in task_sequence:
            # Check capacity: if depleted, we must reload at the depot first
            if curr_load < demand - 1e-6:
                if not destinations or destinations[-1] != self.depot:
                    destinations.append(self.depot)
                curr_load = self.capacity

            # Travel to target node
            if not destinations or destinations[-1] != node:
                destinations.append(node)
            
            # Fulfill demand
            curr_load -= demand

        # Finally, return back to depot to complete the VRP cycle
        if not destinations or destinations[-1] != self.depot:
            destinations.append(self.depot)

        return destinations

    # --- 1. Constructive Initial Heuristic ---
    def _nearest_neighbor_initial(self) -> dict[int, list[tuple[NodeId, int, float]]]:
        """Greedy Nearest-Neighbor task assignment heuristic."""
        solution: dict[int, list[tuple[NodeId, int, float]]] = {t_id: [] for t_id in range(self.num_trucks)}
        unassigned = list(self.tasks)

        # Tracks the current position and remaining load of each truck
        truck_pos = {t_id: self.depot for t_id in range(self.num_trucks)}
        truck_load = {t_id: self.capacity for t_id in range(self.num_trucks)}

        while unassigned:
            for t_id in range(self.num_trucks):
                if not unassigned:
                    break

                curr_pos = truck_pos[t_id]
                curr_load = truck_load[t_id]

                # Filter tasks that fit within remaining capacity
                valid_tasks = [t for t in unassigned if t[2] <= curr_load + 1e-6]

                if not valid_tasks:
                    # Reload at depot
                    truck_pos[t_id] = self.depot
                    truck_load[t_id] = self.capacity
                    continue

                # Find the nearest valid task physically
                next_task = min(valid_tasks, key=lambda t: (self.distances[curr_pos][t[0]], str(t[0]), t[1]))
                
                solution[t_id].append(next_task)
                unassigned.remove(next_task)
                
                truck_pos[t_id] = next_task[0]
                truck_load[t_id] -= next_task[2]

        return solution

    # --- 2. Evaluation Helper ---
    def _evaluate_solution(self, solution: dict[int, list[tuple[NodeId, int, float]]]) -> float:
        """Calculate the total travel distance of the routing solution."""
        total_dist = 0.0
        for t_id, task_seq in solution.items():
            if not task_seq:
                continue
            curr_node = self.depot
            curr_load = self.capacity
            for node, _, demand in task_seq:
                if curr_load < demand - 1e-6:
                    # Return to depot first
                    total_dist += self.distances[curr_node][self.depot]
                    curr_node = self.depot
                    curr_load = self.capacity
                total_dist += self.distances[curr_node][node]
                curr_node = node
                curr_load -= demand
            # Final return to depot
            total_dist += self.distances[curr_node][self.depot]
        return total_dist

    # --- 3. Destroy Operators ---
    def _apply_destroy(
        self,
        solution: dict[int, list[tuple[NodeId, int, float]]],
        op_idx: int,
        n_remove: int,
    ) -> tuple[dict[int, list[tuple[NodeId, int, float]]], list[tuple[NodeId, int, float]]]:
        """Route destruction hub."""
        partial = {k: list(v) for k, v in solution.items()}
        all_tasks = []
        for seq in partial.values():
            all_tasks.extend(seq)

        if len(all_tasks) <= n_remove:
            # If tasks to remove is greater than total, empty everything
            return {t_id: [] for t_id in range(self.num_trucks)}, all_tasks

        removed = []
        if op_idx == 0:
            removed = self._destroy_random(partial, all_tasks, n_remove)
        elif op_idx == 1:
            removed = self._destroy_worst(partial, all_tasks, n_remove)
        elif op_idx == 2:
            removed = self._destroy_shaw(partial, all_tasks, n_remove)

        return partial, removed

    def _destroy_random(
        self,
        partial: dict[int, list[tuple[NodeId, int, float]]],
        all_tasks: list[tuple[NodeId, int, float]],
        n_remove: int,
    ) -> list[tuple[NodeId, int, float]]:
        """Randomly select and remove customer tasks."""
        removed = random.sample(all_tasks, n_remove)
        for t_id in partial:
            partial[t_id] = [task for task in partial[t_id] if task not in removed]
        return removed

    def _destroy_worst(
        self,
        partial: dict[int, list[tuple[NodeId, int, float]]],
        all_tasks: list[tuple[NodeId, int, float]],
        n_remove: int,
    ) -> list[tuple[NodeId, int, float]]:
        """Remove tasks that contribute the highest detour distance costs."""
        removed = []
        for _ in range(n_remove):
            if not all_tasks:
                break
            # Calculate cost contribution delta for each task
            costs = []
            base_cost = self._evaluate_solution(partial)
            for task in all_tasks:
                # Temporary remove
                temp_seq = {t_id: [t for t in partial[t_id] if t != task] for t_id in partial}
                cost_delta = base_cost - self._evaluate_solution(temp_seq)
                costs.append((task, cost_delta))

            # Random bias selection: pick task with probability proportional to its worst-rank
            costs.sort(key=lambda x: (x[1], str(x[0][0]), x[0][1]), reverse=True)
            
            # Select with randomized power bias parameter (p=3)
            idx = int(random.random()**3 * len(costs))
            selected_task = costs[idx][0]
            
            removed.append(selected_task)
            all_tasks.remove(selected_task)
            for t_id in partial:
                partial[t_id] = [t for t in partial[t_id] if t != selected_task]
        return removed

    def _destroy_shaw(
        self,
        partial: dict[int, list[tuple[NodeId, int, float]]],
        all_tasks: list[tuple[NodeId, int, float]],
        n_remove: int,
    ) -> list[tuple[NodeId, int, float]]:
        """Remove tasks that are physically clustered together."""
        seed = random.choice(all_tasks)
        removed = [seed]
        all_tasks.remove(seed)
        for t_id in partial:
            partial[t_id] = [t for t in partial[t_id] if t != seed]

        for _ in range(n_remove - 1):
            if not all_tasks:
                break
            # Find the most similar task to the average of already removed tasks
            similarities = []
            for task in all_tasks:
                avg_dist = np.mean([self.distances[task[0]][r[0]] for r in removed])
                similarities.append((task, avg_dist))

            similarities.sort(key=lambda x: (x[1], str(x[0][0]), x[0][1]))
            
            # Apply randomized power bias selection
            idx = int(random.random()**3 * len(similarities))
            selected_task = similarities[idx][0]
            
            removed.append(selected_task)
            all_tasks.remove(selected_task)
            for t_id in partial:
                partial[t_id] = [t for t in partial[t_id] if t != selected_task]

        return removed

    # --- 4. Repair Operators ---
    def _apply_repair(
        self,
        partial: dict[int, list[tuple[NodeId, int, float]]],
        op_idx: int,
        removed: list[tuple[NodeId, int, float]],
    ) -> dict[int, list[tuple[NodeId, int, float]]]:
        """Route repair/insertion hub."""
        candidate = {k: list(v) for k, v in partial.items()}
        if op_idx == 0:
            candidate = self._repair_greedy(candidate, removed)
        elif op_idx == 1:
            candidate = self._repair_regret(candidate, removed)
        return candidate

    def _repair_greedy(
        self,
        solution: dict[int, list[tuple[NodeId, int, float]]],
        removed: list[tuple[NodeId, int, float]],
    ) -> dict[int, list[tuple[NodeId, int, float]]]:
        """Re-insert tasks into positions minimizing extra travel distance."""
        for task in removed:
            best_t_id, best_idx, _ = self._find_best_insertion(solution, task)
            solution[best_t_id].insert(best_idx, task)
        return solution

    def _repair_regret(
        self,
        solution: dict[int, list[tuple[NodeId, int, float]]],
        removed: list[tuple[NodeId, int, float]],
    ) -> dict[int, list[tuple[NodeId, int, float]]]:
        """Re-insert tasks based on the regret of second-best choices."""
        tasks_to_insert = list(removed)
        while tasks_to_insert:
            regrets = []
            for task in tasks_to_insert:
                # Find best and second-best insertion options
                options = []
                for t_id in range(self.num_trucks):
                    for idx in range(len(solution[t_id]) + 1):
                        solution[t_id].insert(idx, task)
                        cost = self._evaluate_solution(solution)
                        solution[t_id].pop(idx)
                        options.append((t_id, idx, cost))
                options.sort(key=lambda x: (x[2], x[0], x[1]))
                
                best_cost = options[0][2]
                second_best_cost = options[1][2] if len(options) > 1 else best_cost * 1.5
                regret = second_best_cost - best_cost
                regrets.append((task, options[0][0], options[0][1], regret))

            # Select the task with the highest regret first
            regrets.sort(key=lambda x: (x[3], str(x[0][0]), x[0][1], x[1], x[2]), reverse=True)
            best_task, t_id, idx, _ = regrets[0]
            
            solution[t_id].insert(idx, best_task)
            tasks_to_insert.remove(best_task)

        return solution

    def _find_best_insertion(
        self,
        solution: dict[int, list[tuple[NodeId, int, float]]],
        task: tuple[NodeId, int, float],
    ) -> tuple[int, int, float]:
        """Find the best truck ID and index for task insertion."""
        best_cost = float("inf")
        best_t_id = 0
        best_idx = 0

        for t_id in range(self.num_trucks):
            for idx in range(len(solution[t_id]) + 1):
                solution[t_id].insert(idx, task)
                cost = self._evaluate_solution(solution)
                solution[t_id].pop(idx)
                if cost < best_cost:
                    best_cost = cost
                    best_t_id = t_id
                    best_idx = idx

        return best_t_id, best_idx, best_cost

    # --- 5. Adaptive Tuning Engine ---
    def _roulette_wheel_select(self, weights: np.ndarray) -> int:
        """Select an index proportional to weights."""
        total = np.sum(weights)
        r = random.uniform(0, total)
        curr = 0.0
        for idx, w in enumerate(weights):
            curr += w
            if r <= curr:
                return idx
        return len(weights) - 1

    def _update_adaptive_weights(self) -> None:
        """Apply score decay and update selection probabilities."""
        # Update destroy weights
        for i in range(len(self.destroy_weights)):
            if self.destroy_counts[i] > 0:
                avg_score = self.destroy_scores[i] / self.destroy_counts[i]
                self.destroy_weights[i] = self.decay * self.destroy_weights[i] + (1.0 - self.decay) * avg_score
                # Reset counters
                self.destroy_scores[i] = 0.0
                self.destroy_counts[i] = 0

        # Update repair weights
        for j in range(len(self.repair_weights)):
            if self.repair_counts[j] > 0:
                avg_score = self.repair_scores[j] / self.repair_counts[j]
                self.repair_weights[j] = self.decay * self.repair_weights[j] + (1.0 - self.decay) * avg_score
                # Reset counters
                self.repair_scores[j] = 0.0
                self.repair_counts[j] = 0
