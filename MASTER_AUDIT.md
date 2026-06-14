# MASTER_AUDIT.md (The Global Triage List)

This document is the centralized landing zone for all issues discovered by the Critic Agent.
The Critic will append bugs, performance bottlenecks, and architectural flaws here domain by domain.
The Planner and the Human will triage this list, move approved fixes into `TASK.md`, and cross them out here.

---

## ~~Domain: Simulation Engine (`src/env/`)~~ [TRIAGED & COMPLETED]
### Bug Fixes

1. `src/env/multi_agent.py`, Line 159
   - **Reason**: Violates the "Perfect Determinism" dogma. `env.graph.edges` behaves as an unordered set/dict view. When the list of active routes is empty, it falls back to iterating over all edges. The max score tie-breaker relies on insertion order, meaning identical runs could diverge.
   - **Suggested Fix**: Wrap the fallback edge list in `sorted()` to ensure deterministic evaluation.
     ```python
     candidates = sorted(list(env.graph.edges))
     ```

2. `src/env/multi_agent.py`, Lines 133-138
   - **Reason**: Violates the "Crash-Proof Topology" dogma. If `NearestDemandProtagonist` selects demand nodes that belong to a physically disconnected component, `distances.get` defaults to `inf`, but it will still pick `candidates[0]`. DisPATCHING a truck to an unreachable node crashes `GraphEnv` with `nx.NetworkXNoPath`.
   - **Suggested Fix**: Filter candidates by connected components using the environment's `node_to_component` mapping before attempting distance calculations.
     ```python
     reachable_candidates = [
         c for c in candidates 
         if env.node_to_component.get(c) == env.node_to_component.get(truck.current_node)
     ]
     if not reachable_candidates:
         actions[truck_id] = truck.home_depot
         continue
     distances = nx.single_source_dijkstra_path_length(env.graph, truck.current_node, weight="distance")
     destination = min(reachable_candidates, key=lambda node: distances.get(node, float("inf")))
     ```

### Performance Fixes

1. `src/env/graph_env.py`, Line 133
   - **Reason**: Violates the "O(1) Computations" dogma. `_k_hop_edges` uses a standard Python list as a queue for its breadth-first search and calls `queue.pop(0)`. Popping from the front of a list is an O(N) operation, causing the initialization step to scale quadratically with graph size.
   - **Suggested Fix**: Import and use `collections.deque` for O(1) removals.
     ```python
     import collections
     queue = collections.deque([(node, 0)])
     # ...
     curr, depth = queue.popleft()
     ```

2. `src/env/smdp_wrapper.py`, Lines 312-316
   - **Reason**: Violates the "O(1) Computations" dogma. Inside `protagonist_action_mask`, it iterates over all `self.env.trucks.items()` repeatedly *inside* a nested loop over all `self.env.graph.nodes()`. This results in an extremely expensive $O(|Trucks| \times |V|)$ loop running on every decision tick.
   - **Suggested Fix**: Precompute a global dictionary of targeted demand loads *before* the node loop begins to allow O(1) dictionary lookups inside the loop.
     ```python
     # Before iterating over self.env.graph.nodes:
     targeted_loads = {}
     for t in self.env.trucks.values():
         if t.destination is not None:
             targeted_loads[t.destination] = targeted_loads.get(t.destination, 0.0) + t.load

     # Inside the node loop:
     other_targeted = targeted_loads.get(n, 0.0)
     if truck.destination == n:
         other_targeted -= truck.load
     ```

3. `src/env/graph_env.py`, Lines 232-233
   - **Reason**: Violates the "O(1) Computations" dogma. The `observe()` function executes a massive `v.copy()` across every single node and edge dictionary. Since `env.step()` invokes `observe()` on every tick, this creates catastrophic object allocation and garbage collection overhead in the hot-path.
   - **Suggested Fix**: Remove the eager `.copy()` loop.
     ```python
     "nodes": self._obs_nodes,
     "edges": self._obs_edges,
     ```

## ~~Domain: RL Brains (`src/agents/`)~~ [TRIAGED & COMPLETED]
### Bug Fixes

1. `src/agents/networks.py`, Line 48
   - **Reason**: Violates the "Perfect Determinism" dogma. `nodes_dict.keys()` produces an insertion-ordered sequence that can vary across identical topological states if the environment rebuilds the graph. This results in non-deterministic `node_to_idx` mappings, causing cache misses and varying PyG `edge_index` generation, which diverges training mathematically.
   - **Suggested Fix**: Wrap the keys in `sorted(list(...))` to enforce strict ordering.
     ```python
     node_ids = sorted(list(nodes_dict.keys()))
     ```

2. `src/agents/networks.py`, Lines 66 and 142
   - **Reason**: The cached `norm_dists` array relies on `edges_dict.items()` iterating in the exact same order during every cache hit. If the environment rebuilds the dictionary in a different order, the loop at Line 142 will silently pair the new `congestion_level` values with the wrong static `norm_dists` precomputed in Line 66, corrupting the edge features tensor.
   - **Suggested Fix**: Sort the edge keys before iterating in both the caching step and the feature-building step to guarantee alignment.
     ```python
     sorted_edges = sorted(list(edges_dict.keys()))
     for edge in sorted_edges:
         u, v = edge
         edata = edges_dict[edge]
     ```

3. `src/agents/sacred_atla.py`, Lines 149-150
   - **Reason**: In the transition builder, `action=actions` assigns a reference to the mutable `actions` dictionary. Since `actions` accumulates the routing choices for *all* trucks over the entire epoch loop, all transitions from that loop will incorrectly share a pointer to the same fully-populated dictionary rather than preserving isolated state snapshots for replay.
   - **Suggested Fix**: Save a shallow copy of the dictionary at that specific time step.
     ```python
     action=dict(actions),  # Save a copy of the current actions
     ```

4. `src/agents/sac.py`, Lines 797-798 and 857-858
   - **Reason**: Cost budget masking zeroes out congested level probabilities. If only the 'wait' action remains valid, and its unmasked softmax output underflows near `0.0`, dividing by `torch.sum(next_flat_probs)` (or `flat_probs`) triggers a division by zero. This results in `NaN` tensors that instantly destroy the network's weights during backpropagation.
   - **Suggested Fix**: Add an explicit check against an empty probability sum and default to the wait action.
     ```python
     # Line 797 & 857 pattern
     sum_probs = torch.sum(flat_probs)
     if sum_probs < 1e-8:
         flat_probs = torch.zeros_like(flat_probs)
         flat_probs[-1] = 1.0
     else:
         flat_probs = flat_probs / sum_probs
     ```

### Performance Fixes

1. `src/agents/sacred_atla.py`, Lines 104-106
   - **Reason**: Violates the "O(1) Computations" dogma. The code calls `copy.deepcopy(event.observation)` inside the decision hot-path loop for every active truck. Deep copying standard python dicts is an exceptionally slow O(N) operation, causing a massive performance bottleneck during stepping.
   - **Suggested Fix**: Replace it with targeted shallow copying of only the nested dictionary that actually needs mutation (`trucks`).
     ```python
     actions = {}
     projected_obs = dict(event.observation)
     projected_obs["trucks"] = {k: dict(v) for k, v in event.observation["trucks"].items()}
     ```

## ~~Domain: Strategic Planners (`src/sbo/`)~~ [TRIAGED & COMPLETED]

### Bug Fixes

1. `src/sbo/flp_solver.py`, Lines 49-53
   - **Reason**: The code hardcodes distributions for exactly `num_depots=2` and `num_trucks=3` (`[pair[0], pair[0], pair[1]]`, etc.), completely ignoring the passed arguments for `num_trucks` and `num_depots`. If `num_depots` is anything other than 2, this raises an `IndexError` or produces invalid distributions, and it fails to assign all `num_trucks` trucks properly.
   - **Suggested Fix**: Use `itertools.combinations_with_replacement` to correctly assign the remaining `num_trucks - num_depots` trucks across the chosen `num_depots`.
     ```python
     allocations = []
     for depots in itertools.combinations(self.node_list, num_depots):
         for extra_trucks in itertools.combinations_with_replacement(depots, num_trucks - num_depots):
             allocations.append(list(depots) + list(extra_trucks))
     ```

2. `src/sbo/surrogate.py`, Line 76
   - **Reason**: Violates the "Perfect Determinism" dogma. `DataLoader(shuffle=True)` is used without a fixed `generator`. This causes the dataset to be shuffled randomly based on the global PyTorch state, leading to non-deterministic training outcomes across identical runs.
   - **Suggested Fix**: Initialize a deterministic random number generator for the DataLoader.
     ```python
     generator = torch.Generator(device=device)
     generator.manual_seed(42)  # Or pass seed dynamically
     loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
     ```

### Performance Fixes

1. `src/sbo/flp_solver.py`, Lines 56-61
   - **Reason**: Violates the "O(1) Computations" dogma. Inside the inner loop, `demands` is unnecessarily recomputed O(N) times even though it doesn't change. Furthermore, running a forward pass on `self.model(features)` sequentially for every individual allocation creates a massive performance bottleneck.
   - **Suggested Fix**: Hoist the constant `demands` list out of the loop and batch the feature tensors to compute predictions in a single forward pass.
     ```python
     demands = [float(demand_dict.get(node, 0.0)) for node in self.node_list]
     feature_list = []
     for alloc in allocations:
         truck_counts = [float(alloc.count(node)) for node in self.node_list]
         feature_list.append(truck_counts + demands)
     
     with torch.no_grad():
         features_tensor = torch.tensor(feature_list, dtype=torch.float32)
         predicted_costs = self.model(features_tensor).squeeze(-1)
         best_idx = torch.argmin(predicted_costs).item()
         
         best_cost = float(predicted_costs[best_idx].item())
         best_allocation = allocations[best_idx]
     ```

## Domain: Baselines (`src/baselines/`)

### Bug Fixes

1. `src/baselines/metaheuristic.py`, Lines 57-60
   - **Reason**: Violates "Perfect Determinism" (iterating over an unordered NetworkX nodes view) and "Crash-Proof Topology" (failing to filter out nodes disconnected from the depot, which will crash `self.distances[curr_node][node]` with a `KeyError`).
   - **Suggested Fix**: Iterate over a sorted list of nodes and filter out any nodes that don't share the depot's connected component.
     ```python
     for node in sorted(list(env.graph.nodes)):
         data = env.graph.nodes[node]
         if data["has_depot"] or data["demand"] <= 0:
             continue
         if env.node_to_component.get(node) != env.node_to_component.get(self.depot):
             continue
     ```

2. `src/baselines/metaheuristic.py`, Lines 228, 320, 354, 411, 419
   - **Reason**: Violates "Perfect Determinism". Using `min()` or `sort()` solely on floating-point values (`distance`, `cost_delta`, `avg_dist`, `regret`) leaves tie-breakers to insertion order. On symmetric graphs with equal distances, this guarantees divergent routing behavior across environments.
   - **Suggested Fix**: Add deterministic secondary tie-breaking elements (like the task tuple or truck indices) to the key functions.
     ```python
     # Line 228
     next_task = min(valid_tasks, key=lambda t: (self.distances[curr_pos][t[0]], t))
     # Line 320
     costs.sort(key=lambda x: (x[1], x[0]), reverse=True)
     # Line 354
     similarities.sort(key=lambda x: (x[1], x[0]))
     # Line 411
     options.sort(key=lambda x: (x[2], x[0], x[1]))
     # Line 419
     regrets.sort(key=lambda x: (x[3], x[0]), reverse=True)
     ```

### Performance Fixes

1. `src/baselines/metaheuristic.py`, Lines 406-410 and 438-442
   - **Reason**: Violates the "O(1) Computations" dogma. In `_find_best_insertion` and `_repair_regret`, `copy.deepcopy(solution)` is called inside doubly-nested loops over all trucks and all insertion indices. This creates catastrophic object allocation overhead, completely bottlenecking the repair operators.
   - **Suggested Fix**: Mutate the `solution` list in-place to evaluate the cost, then immediately revert the change using `.pop(idx)`.
     ```python
     # For _find_best_insertion:
     for idx in range(len(solution[t_id]) + 1):
         solution[t_id].insert(idx, task)
         cost = self._evaluate_solution(solution)
         solution[t_id].pop(idx)
         if cost < best_cost:
             best_cost = cost
             best_t_id = t_id
             best_idx = idx
             
     # Similar fix applies to _repair_regret:
     solution[t_id].insert(idx, task)
     cost = self._evaluate_solution(solution)
     solution[t_id].pop(idx)
     options.append((t_id, idx, cost))
     ```

2. `src/baselines/metaheuristic.py`, Lines 89, 114, 116, 121, 129, 268, 375
   - **Reason**: Violates the "O(1) Computations" dogma. `copy.deepcopy()` is used repeatedly throughout the ALNS solver (e.g., `best_sol = copy.deepcopy(current_sol)`) to duplicate dictionary-of-lists routing states. Python's `copy.deepcopy` is notoriously slow and unnecessary here since the inner task elements are immutable tuples.
   - **Suggested Fix**: Replace all `copy.deepcopy(solution)` calls with a shallow dictionary comprehension that copies the lists.
     ```python
     best_sol = {k: list(v) for k, v in candidate_sol.items()}
     # Apply equivalent list comprehension to all other deepcopy calls
     ```
