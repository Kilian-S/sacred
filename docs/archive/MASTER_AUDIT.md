# MASTER_AUDIT.md (The Global Triage List)

This document is the centralized landing zone for all issues discovered by the Critic Agent.
The Critic will append bugs, performance bottlenecks, and architectural flaws here domain by domain.
The Planner and the Human will triage this list, move approved fixes into `TASK.md`, and cross them out here.

> **UNAUDITED (added since the last critic pass, 2026-06-28/29) — candidates for the next review:**
> `src/agents/transition_builder.py` (shared protagonist transition builder — the single source of truth for projection + sequential claiming; has a format-drift guard test), `src/envs/assignment_factory.py`, `scripts/{evaluate_assignment,run_generation,aggregate_generation,generate_erb_assign}.py`, and the new `--seed/--group/--threads/--erb-path` paths + `assign` branch in `scripts/train_sacred.py`. Earlier-found held items (low severity) are summarised in `CONTEXT.md` §2 "Held / known issues". The A\* `_heuristic` inadmissibility was FIXED (routing + greedy ETAs now use exact Dijkstra).

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

## ~~Domain: Baselines (`src/baselines/`)~~ [VERIFIED FIXED IN CODE 2026-06-27]

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

## ~~Domain: SMDP Wrapper (`src/env/smdp_wrapper.py`)~~ [VERIFIED FIXED IN CODE 2026-06-27]

### Bug Fixes

1. `src/env/smdp_wrapper.py`, Line 354
   - **Reason**: Violates the "Perfect Determinism" dogma. `nearby_edges` is an unordered python `set()`. Iterating over it directly causes the `levels_by_edge` dictionary to be populated in a non-deterministic insertion order, which will cause diverging action masks and diverging mathematical results for identical seeds across runs.
   - **Suggested Fix**: Wrap the set in a sorted list before iterating over it.
     ```python
     for edge in sorted(list(nearby_edges)):
     ```

2. `src/env/smdp_wrapper.py`, Lines 300-307
   - **Reason**: Violates the "Crash-Proof Topology" dogma. If a truck has no load (or remaining demand is zero) and needs to return to the depot, it forcefully adds `[truck.home_depot]` to the action mask. If the truck's current node is physically disconnected from the depot, dispatching the truck will crash the physics engine with `nx.NetworkXNoPath`.
   - **Suggested Fix**: Add a component check before appending the depot, defaulting to an empty list if unreachable.
     ```python
     if truck.load <= 0 and current_node != truck.home_depot:
         if self.env.node_to_component.get(current_node) == self.env.node_to_component.get(truck.home_depot):
             mask[truck_id] = [truck.home_depot]
         else:
             mask[truck_id] = []
     elif truck.load <= 0:
         mask[truck_id] = []
     elif self._remaining_demand() <= 0 and current_node != truck.home_depot:
         if self.env.node_to_component.get(current_node) == self.env.node_to_component.get(truck.home_depot):
             mask[truck_id] = [truck.home_depot]
         else:
             mask[truck_id] = []
     ```

3. `src/env/smdp_wrapper.py`, Line 239
   - **Reason**: Logic Bug. In `step_antagonist`, if the antagonist enters a sequential epoch (`continue_loop == True`), the `SMDPTransition` explicitly hardcodes `reward=0.0`. However, a `cost_penalty` was just subtracted from `_accumulated_antagonist_reward`, which gets packaged into `next_event` and then the accumulators are reset. This means the intermediate congestion costs are permanently lost from the replay buffer transitions, preventing the RL agent from learning the cost of its sequential actions.
   - **Suggested Fix**: Assign `next_event.antagonist_reward` to the transition reward instead of hardcoding `0.0`.
     ```python
     reward=next_event.antagonist_reward,
     ```

### Performance Fixes

1. `src/env/smdp_wrapper.py`, Lines 388-392
   - **Reason**: Violates the "O(1) Computations" dogma. Inside `_build_event`, `self.protagonist_action_mask()` is called sequentially twice (once for `waiting_trucks`, once for `protagonist_action_mask`), and it was just called previously in `_current_decision_type()` to trigger the event. Since `protagonist_action_mask()` contains expensive loops, executing it repeatedly for a single event generation causes a massive unnecessary computation spike.
   - **Suggested Fix**: Evaluate the mask once and store it in a local variable before building the `DecisionEvent` object.
     ```python
     prot_mask = self.protagonist_action_mask()
     event = DecisionEvent(
         decision_type=decision_type,
         observation=self.env.observe(),
         waiting_trucks=[truck_id for truck_id, options in prot_mask.items() if options],
         protagonist_reward=self._accumulated_protagonist_reward,
         antagonist_reward=self._accumulated_antagonist_reward,
         elapsed_ticks=self._elapsed_ticks,
         protagonist_action_mask=prot_mask,
         antagonist_action_mask=self.antagonist_action_mask(),
         done=decision_type == DecisionType.TERMINAL,
         info={
     ```

2. `src/env/smdp_wrapper.py`, Lines 312-334
   - **Reason**: Violates the "O(1) Computations" dogma. Inside `protagonist_action_mask`, the code loops over all graph nodes inside the truck loop. While dictionary lookups were hoisted previously, the loop over `self.env.graph.nodes()` itself wasn't. If a truck has no valid destinations, the mask returns empty and no decision event triggers, causing this $O(|Trucks| \times |V|)$ loop to re-execute wastefully on every single simulated second until the episode ends.
   - **Suggested Fix**: Precompute a global list of valid customer demand nodes (grouped by component) outside of the truck loop to bypass the full-graph scan.
     ```python
     # Before the truck loop:
     valid_customers_by_comp = {}
     for n, data in self.env.graph.nodes(data=True):
         if not data.get("has_depot", False) and float(data.get("demand", 0.0)) > 0.0:
             comp = self.env.node_to_component.get(n)
             valid_customers_by_comp.setdefault(comp, []).append((n, float(data.get("demand", 0.0))))
             
     # Inside the truck loop:
     truck_comp = self.env.node_to_component.get(current_node)
     destinations = []
     for n, node_demand in valid_customers_by_comp.get(truck_comp, []):
         other_targeted = targeted_loads.get(n, 0.0)
         if truck.destination == n:
             other_targeted -= truck.load
         if (node_demand - other_targeted) > 0.0:
             destinations.append(n)
     ```

## ~~Domain: Strategic Planners (`src/sbo/`) - New Findings~~ [VERIFIED FIXED IN CODE 2026-06-27]

### Bug Fixes

1. `src/sbo/flp_solver.py`, Line 63
   - **Reason**: Logic Bug / Crash Risk. The PyTorch tensor `features_tensor` is instantiated on the CPU by default, but `self.model` may be loaded on a hardware accelerator (like MPS on Apple Silicon, per SYSTEM.md). Passing a CPU tensor to an accelerated model will trigger a fatal PyTorch device mismatch crash.
   - **Suggested Fix**: Extract the target device from the surrogate model's parameters and create the tensor directly on that device.
     ```python
     device = next(self.model.parameters()).device
     features_tensor = torch.tensor(all_features, dtype=torch.float32, device=device)
     ```

### Performance Fixes

1. `src/sbo/flp_solver.py`, Lines 49-51
   - **Reason**: Violates the "O(1) Computations" dogma. The solver generates many invalid truck allocations using `combinations_with_replacement(pair, num_trucks)` and then discards them via an expensive `if len(set(alloc)) == num_depots:` set-casting check, wasting massive compute.
   - **Suggested Fix**: Generate exclusively valid combinations by assigning one truck per depot explicitly, and distributing only the remaining `num_trucks - num_depots` trucks.
     ```python
     for pair in depot_pairs:
         base_alloc = tuple(pair)
         for extra_alloc in itertools.combinations_with_replacement(pair, num_trucks - num_depots):
             allocations.append(base_alloc + extra_alloc)
     ```

2. `src/sbo/flp_solver.py`, Lines 60-61
   - **Reason**: Violates the "O(1) Computations" dogma. Inside a nested loop over `allocations`, it iterates sequentially over all nodes in `self.node_list` and calls the $O(K)$ operation `alloc.count(node)`. It also performs an expensive list concatenation `truck_counts + demands` repeatedly, duplicating lists and thrashing the garbage collector.
   - **Suggested Fix**: Eliminate the `.count()` bottleneck by iterating only over the items in `alloc`, and eliminate list duplication by writing directly into a pre-allocated PyTorch tensor.
     ```python
     num_nodes = len(self.node_list)
     features_tensor = torch.zeros((len(allocations), 2 * num_nodes), dtype=torch.float32)
     node_to_idx = {node: i for i, node in enumerate(self.node_list)}
     for i, alloc in enumerate(allocations):
         for node in alloc:
             features_tensor[i, node_to_idx[node]] += 1.0
             
     demands = [float(demand_dict.get(n, 0.0)) for n in self.node_list]
     features_tensor[:, num_nodes:] = torch.tensor(demands, dtype=torch.float32)
     ```

## ~~Domain: RL Brains (`src/agents/networks.py`) - New Findings~~ [VERIFIED FIXED IN CODE 2026-06-27]

> Note: bug #2 (edge permutation-invariance) is moot in this pipeline — observation edges are
> pre-canonicalized by `GraphEnv._edge_key` (`(u,v)` with `repr(u)<=repr(v)`), so the antagonist
> net only ever sees one consistent ordering. bug #1 (`node_in_dim=9`) and #3 (edge-logit masking
> when no level is affordable) are implemented.

### Bug Fixes

1. `src/agents/networks.py`, Lines 169-170, 226-227, 304-305
   - **Reason**: PyTorch tensor shape mismatch. The `node_in_dim` default across `GATv2Encoder`, `ProtagonistPolicyValueNet`, and `AntagonistPolicyValueNet` is statically set to `7`. However, the `featurize_state` function (Lines 124-135) explicitly constructs a node feature vector of length `9` (`[x_norm, y_norm, demand, is_depot, num_trucks, is_active_here, active_load, is_targeted, unassigned]`). Passing the size-9 `pyg_data.x` to the size-7 encoder will trigger an immediate dimension mismatch crash in `GATv2Conv`.
   - **Suggested Fix**: Update the default `node_in_dim` argument to `9` across all network initializers.
     ```python
     node_in_dim: int = 9,
     ```

2. `src/agents/networks.py`, Line 405
   - **Reason**: Mathematical correctness breach (Permutation Invariance). In `AntagonistPolicyValueNet`, the `combined` edge tensor is created by concatenating `[emb_u, emb_v, attr]`. Because `edge_mlp` applies linear weights, it will assign entirely different weights to the first and second half of the node embeddings. Consequently, the network's evaluation of an undirected edge becomes direction-dependent (`score(u, v) != score(v, u)`), violating the symmetric nature of undirected edges and destabilizing learning when the environment yields randomly ordered node tuples.
   - **Suggested Fix**: Enforce a deterministic ordering of the node embeddings before concatenation (e.g., by node index) so the MLP always evaluates `[emb_min, emb_max, attr]`.
     ```python
     if idx_u > idx_v:
         emb_u, emb_v = emb_v, emb_u
     combined = torch.cat([emb_u, emb_v, attr], dim=-1)
     ```

3. `src/agents/networks.py`, Lines 438-440
   - **Reason**: Action masking flaw. In `AntagonistPolicyValueNet`, if `remaining_budget` is too low to afford even the cheapest congestion level, `level_mask` evaluates to entirely `False`. The code safely masks `level_logits` with `-1e9`, causing `level_probs` to degrade into a uniform distribution `[0.25, ...]`. Crucially, it fails to concurrently mask `edge_logits`. This allows the network to select an edge to congest, and then randomly sample an invalid, unaffordable congestion level, violating strict budget constraints.
   - **Suggested Fix**: Check if `level_mask` is completely empty. If so, immediately mask out all `edge_logits` to force the network to choose the `wait` action.
     ```python
     if not level_mask.any():
         edge_logits = torch.full_like(edge_logits, -1e9)
     ```

## ~~Physics & Performance~~ [VERIFIED FIXED IN CODE 2026-06-27]

> All items below verified implemented: vectorized truck movement, O(1) `_idle_trucks_at_depot`
> in `is_done`, batched A* cache-clear + precomputed `effective_weight`, tuple paths, multi_agent
> `_all_pairs_distances` + vectorized `np.hypot` gravity + tracked `total_remaining_demand`/
> `expected_demand`, congestion heap, `is_idle` early-exit, and global `valid_customers_by_comp`.

1. `src/env/graph_env.py`, Line 423
   - **Reason**: Violates the "O(1) Computations" dogma. The `while remaining_time > 1e-12` loop in `_move_truck_one_tick` (called in a `for` loop over all trucks at line 221) calculates physics step-by-step for each truck individually in Python. This unvectorized Python while/for loop in the hot-path scales poorly with the number of trucks and time resolution.
   - **Suggested Fix**: Vectorize the physics update using NumPy arrays for truck states (positions, edge progress) to compute all truck movements in a single step.

2. `src/env/graph_env.py`, Line 263
   - **Reason**: Violates the "O(1) Computations" dogma. Despite the comment `# O(1) check if all trucks are at a depot`, iterating `for truck in self.trucks.values():` is an O(N) operation proportional to the number of trucks. Called every tick inside `is_done()`, this slows down the hot path.
   - **Suggested Fix**: Maintain an integer counter (e.g., `self._idle_trucks_at_depot`) that increments when a truck arrives at a depot and decrements when dispatched. Return a true O(1) dictionary/set check: `self.remaining_demand <= 0 and self._idle_trucks_at_depot == self.num_trucks`.

3. `src/env/graph_env.py`, Lines 280, 304, and 506
   - **Reason**: Inefficient graph algorithms and A* cache invalidations. `self._get_shortest_path.cache_clear()` is called for *every* individual edge congestion update (Line 280), completely thrashing the A* cache. Furthermore, `nx.astar_path` uses a slow local Python callable `weight_func` (Line 304) and a `_heuristic` with localized imports and slow NetworkX dictionary lookups (Line 506).
   - **Suggested Fix**: Batch cache invalidations to clear at most once per tick. Precompute `effective_weight` as an edge attribute to avoid the Python `weight_func` overhead, and vectorize/precompute coordinate lookups for the heuristic.

4. `src/env/graph_env.py`, Line 243
   - **Reason**: Memory allocation bottlenecks (object churn). Calling `list(truck.path)` inside `observe()` allocates a new Python list object for every truck on every single tick. In an RL hot-path, this causes massive object churn and heavy garbage collection overhead.
   - **Suggested Fix**: Store paths as immutable tuples instead of lists so they can be referenced directly in the observation dictionary without deepcopying or re-allocating.

5. `src/env/multi_agent.py`, Lines 134-138
   - **Reason**: Inefficient graph algorithms (nx overhead). `nx.single_source_dijkstra_path_length` is invoked inside a for-loop over idle trucks on every tick. This evaluates a very expensive unvectorized shortest-path search repeatedly in the hot-path instead of relying on precomputed values, heavily degrading tick speed.
   - **Suggested Fix**: Precompute an all-pairs shortest path matrix (via Floyd-Warshall or cached Dijkstra) during graph initialization. Replace the `nx` call with an O(1) dictionary or array lookup (`self.distance_matrix[current_node][candidate]`).

6. `src/env/multi_agent.py`, Lines 188-195
   - **Reason**: Unvectorized Python while/for loops. In `_downstream_demand_gravity`, a raw Python loop over `env.graph.nodes(data=True)` computes `hypot` distances sequentially. If the antagonist evaluates all edges, this produces a catastrophic $O(|E| \times |V|)$ nested Python loop per tick.
   - **Suggested Fix**: Vectorize coordinates and demands using NumPy arrays. Compute the midpoint distances and gravity scores for all candidate edges simultaneously via broadcasted NumPy tensor operations.

7. `src/env/multi_agent.py`, Lines 341-342
   - **Reason**: O(N) operations that should be O(1). `_remaining_demand()` calculates total demand by iterating through all nodes in `self.env.graph.nodes(data=True)`. Since it is called by `_agent_rewards` and `_update_metrics` on every tick, it causes a constant $O(|V|)$ overhead.
   - **Suggested Fix**: Track `total_remaining_demand` globally within the environment object state. Decrement it in O(1) whenever a delivery event takes place, and return the tracked value directly.

8. `src/env/multi_agent.py`, Lines 100-102
   - **Reason**: Memory allocation bottlenecks / O(N) operations that should be O(1). `NearestDemandProtagonist` rebuilds the `expected_demand` dictionary from scratch every tick by looping over all graph nodes, causing heavy object churn and an $O(|V|)$ allocation penalty.
   - **Suggested Fix**: Maintain an active dictionary of nodes with remaining demand globally. Update it incrementally in O(1) only when trucks are dispatched or deliveries occur to avoid the full-graph iteration.

9. `src/env/smdp_wrapper.py`, Lines 379-383
   - **Reason**: O(N) operations that should be O(1). In `_current_decision_type`, `self.protagonist_action_mask()` is evaluated on every single simulated tick inside the `advance_until_decision` loop. This forces expensive $O(|V|)$ loop scans and dictionary allocations on every second, even when no trucks are idle, creating a massive hot-path bottleneck.
   - **Suggested Fix**: Add an O(1) early exit in `_current_decision_type` or at the top of `protagonist_action_mask`: `if not any(t.is_idle for t in self.env.trucks.values()): return False` (or return `{}`).

10. `src/env/smdp_wrapper.py`, Lines 198-203
   - **Reason**: Unvectorized Python while/for loops. The GPS-style dynamic rerouting uses a doubly-nested Python loop (over all trucks, then over all edges in each truck's path) to check for congested edges. It repeatedly calls `self.env._edge_key(u, v)` inside the inner loop, creating an $O(|E_{congested}| \times |Trucks| \times PathLength)$ overhead.
   - **Suggested Fix**: Cache a normalized edge set on the truck object when its path is assigned (e.g., `truck.path_edges = set(...)`). This converts the inner path-scan into an O(1) lookup: `if edge in truck.path_edges:`.

11. `src/env/smdp_wrapper.py`, Lines 484-491
   - **Reason**: Inefficient graph algorithms (nx overhead). `_neighbors_toward` invokes `nx.shortest_path(self.env.graph, ...)` directly. This evaluates a full Dijkstra search from scratch on every call, bypassing environment A* caching, causing an $O(|E| + |V| \log |V|)$ graph algorithm bottleneck.
   - **Suggested Fix**: Replace the slow NetworkX call with an O(1) lookup against a precomputed all-pairs shortest-path matrix, or route the call through the environment's cached pathfinder (`self.env._get_shortest_path`).

12. `src/env/smdp_wrapper.py`, Lines 465-472
   - **Reason**: Unvectorized Python while/for loops. The `_age_congestion` method iterates through the `self.active_congestion` dictionary and decrements integer ticks on *every single simulated second*. This is an unvectorized O(N) update loop in the physics hot-path causing unnecessary memory writes.
   - **Suggested Fix**: Store absolute expiration ticks (`expiration_tick = env.time + duration`) instead of remaining ticks. Use a minimum heap (priority queue) to achieve O(1) checks for expired edges, eliminating the per-tick loop.

13. `src/env/smdp_wrapper.py`, Lines 294-302
   - **Reason**: Memory allocation bottlenecks (object churn). `valid_customers_by_comp` allocates new lists and tuples dynamically by iterating over `self.env.graph.nodes(data=True)`. Even on valid decision ticks, this builds redundant $O(|V|)$ objects that are instantly garbage collected, creating heavy object churn.
   - **Suggested Fix**: Precompute `valid_customers_by_comp` globally in the environment state and update it incrementally in O(1) only when demands change, rather than rebuilding it dynamically inside the action mask function.

## Domain: SMDP Wrapper (`src/env/smdp_wrapper.py`) - Domain Logic Critic Findings [2026-06-27]

*Scope: mathematical correctness, SMDP/SAC logic, action masking, and the "Perfect Determinism" / "Crash-Proof Topology" dogmas. Performance is excluded by mandate. Findings below are NEW — distinct from the three already triaged in the section above (sorted edges, depot reachability, antagonist sequential reward). This file is pure-Python physics with no torch tensors, so the "shape/dtype" bug class is N/A here.*

### Bug Fixes

1. `src/env/smdp_wrapper.py`, Lines 580-602 (accumulation) and 646-647 (`smdp_discount`) — **SMDP intra-option rewards are not discounted (mathematical inexactness in the SMDP Bellman backup).**
   - **Reason**: `_accumulate_step` builds the option reward as a *plain undiscounted sum* of the per-tick rewards (`self._accumulated_protagonist_reward += protagonist_reward`), and the only discounting available to the agent is `smdp_discount = gamma ** elapsed_ticks`, applied to the *bootstrap* term `Q(s')`. The exact SMDP backup is `Q(s,a) = E[ Σ_{k=0}^{τ-1} γ^k r_{t+k} + γ^τ Q(s',a') ]`, i.e. the intra-option rewards must each be discounted by `γ^k`. The current code instead treats all `τ` rewards as if received at the decision instant `t` (effectively `γ^0`). For `reward_mode="latency"`, where every tick contributes `-remaining_demand` and an option can span `antagonist_interval`≈20–40 ticks, this systematically *under-discounts* the within-option penalty: with `γ=0.99`, `γ^20≈0.82`, so the last ticks of a long option are over-weighted relative to the true objective. The value targets are therefore biased (not merely scaled), which distorts the relative ranking of short-vs-long options — exactly the latency tradeoff the policy is meant to learn.
   - **Note**: This may be a deliberate "lumped-reward" approximation (very common in option/SMDP code, and the lump cannot be re-discounted later because per-tick rewards are not retained). Flagging because it is a real departure from the SMDP math and interacts directly with the latency objective. Once lumped, it is unrecoverable downstream — the fix must live at accumulation time.
   - **Suggested Fix**: Discount at accumulation time, carrying a per-option tick counter. This keeps `γ` out of the physics engine (the wrapper already owns RL reward shaping per the dogma), but `γ` must be threaded into `SMDPConfig` or `_accumulate_step`:
     ```python
     # in _accumulate_step, with self._option_tick starting at 0 in _reset_accumulators:
     discount = self.config.gamma ** self._option_tick
     self._accumulated_protagonist_reward += discount * protagonist_reward
     self._accumulated_antagonist_reward  += discount * antagonist_reward
     self._option_tick += 1
     ```
     The agent then still bootstraps with `gamma ** elapsed_ticks` and the two halves compose into the exact SMDP backup. If the lumped approximation is intentional, document it explicitly at line 646 so it is not silently relied upon as exact.

2. `src/env/smdp_wrapper.py`, Lines 192-197 (`step_protagonist`) vs Lines 321-325 (`advance_until_decision`) — **The protagonist dispatch tick advances simulated time by one second but never calls `_age_congestion`, unlike every other tick.**
   - **Reason**: In `advance_until_decision`, the canonical tick loop is `step → _accumulate_step → _age_congestion → _auto_resolve_forced_moves`, so every simulated second ages congestion exactly once. But in `step_protagonist`, the dispatch step `self.env.step(...)` (line 193/195) is followed only by `_accumulate_step` (line 196) and then `advance_until_decision`, whose *immediate* return path (lines 316-319) can return a decision event **before** the loop body — i.e. without ever aging that tick. Consequently the single simulated second consumed by a protagonist dispatch is skipped by the aging logic: congestion entries whose `expiration_tick` equals the post-dispatch `env.time` survive one extra tick, and `cooldown_remaining` is not decremented for that tick. The effect is a systematic off-by-one in congestion *duration* and *cooldown* that fires precisely on protagonist decision boundaries. It is deterministic (so not a determinism-dogma breach), but it makes the antagonist's congestion last marginally longer than `congestion_duration` advertises, and is correlated with protagonist activity — a subtle bias in the adversarial dynamics.
   - **Suggested Fix**: Age the dispatch tick explicitly, mirroring the loop body, so all ticks are treated uniformly:
     ```python
     step_result = self.env.step(...)          # existing
     self._accumulate_step(step_result, antagonist_action={})
     self._age_congestion()                    # ADD: keep tick accounting symmetric
     next_event = self.advance_until_decision()
     ```
     (Verify there is no double-aging on the path where `advance_until_decision`'s loop also runs — the immediate-return guard at line 317 ensures the first loop iteration won't re-age the same `env.time`, so this addition ages each second exactly once.)

3. `src/env/smdp_wrapper.py`, Lines 414-422 (`_next_hop_action_mask`) — **Corridor bound degenerates to "all neighbours" when the goal is unreachable, because `inf <= inf` evaluates True (edge case, low severity).**
   - **Reason**: When `cur = dist_to_goal.get(truck.current_node, inf)` is `inf` (the chosen goal is in a different connected component), `budget = slack * inf + 1e-9 = inf`, and the admission test `via <= budget` becomes `inf <= inf`, which is `True`. Every neighbour is then admitted as "forward", silently disabling the corridor restriction. In practice this is **benign**: if `current_node` cannot reach `goal`, then by the triangle inequality no adjacent neighbour can either, so all `via` are `inf` and the result is identical to the existing `forward if forward else neighbors` fallback (line 422). No crash risk (the move is always to a graph-adjacent node, so `dispatch_truck_edge` cannot raise `NetworkXNoPath`). Flagging only because the `inf <= inf` admission is an accidental, non-obvious code path that future edits could turn into a real corridor leak.
   - **Suggested Fix**: Guard the unreachable-goal case to make the intent explicit and avoid relying on IEEE `inf <= inf` semantics:
     ```python
     if cur == float("inf"):
         mask[truck_id] = neighbors          # goal in another component; corridor undefined
         continue
     budget = self.config.routing_corridor_slack * cur + 1e-9
     ```

## Domain: Simulation Engine (`src/env/graph_env.py`) - Domain Logic Critic Findings [2026-06-27]

*Scope: mathematical correctness, logic bugs, action masking, and the "Perfect Determinism" / "Crash-Proof Topology" dogmas. Performance optimizations are strictly ignored.*

### Bug Fixes

1. `src/env/graph_env.py`, Lines 285-303
   - **Reason**: Mathematical correctness violation (Chronological Time Resolution). The vectorized physics engine processes all trucks that arrive at their destinations within the current tick simultaneously in a single `while True` loop iteration. If Truck A arrives at 0.2s and Truck B arrives at 0.8s, both are flagged as `arrived` and executed sequentially sorted by `truck_id`. If Truck B has a lower ID, it will process its arrival and serve demand *before* Truck A, violating the true temporal ordering of events. Furthermore, if Truck A enters a new edge and arrives again at 0.5s, that second arrival is executed in the *next* loop iteration (i.e. after B's 0.8s arrival). This destroys continuous event sequencing and causes the wrong trucks to consume demand when competing for the same nodes.
   - **Suggested Fix**: Calculate absolute arrival times within the tick for all active trucks. In each iteration, isolate the earliest arrival time, process *only* the truck(s) tying for that minimum time, and leave the remaining trucks for subsequent loop iterations.
     ```python
     # Replace the current arrived/not_arrived execution blocks with:
     arr_times = np.full(n, np.inf)
     active_and_arriving = active_mask & ((max_dists + 1e-12) >= rem_dists)
     if active_and_arriving.any():
         arr_times[active_and_arriving] = (1.0 - rem_times[active_and_arriving]) + (rem_dists[active_and_arriving] / eff_speeds[active_and_arriving])
     
     if not_arrived.any():
         idx = np.where(not_arrived)[0]
         for i in idx:
             active_trucks[i].edge_progress += max_dists[i]
             travelled[i] += max_dists[i]
             rem_times[i] = 0.0
             
     if active_and_arriving.any():
         min_arr_time = np.min(arr_times)
         earliest_idx = np.where(np.abs(arr_times - min_arr_time) < 1e-9)[0]
         # Deterministic tie-breaker for identical arrival times
         sorted_idx = sorted(earliest_idx, key=lambda x: active_trucks[x].truck_id)
         for i in sorted_idx:
             t = active_trucks[i]
             u, v = t.edge
             edge_dist = self.graph.edges[u, v]["distance"]
             t.edge_progress = edge_dist
             travelled[i] += rem_dists[i]
             rem_times[i] -= rem_dists[i] / eff_speeds[i]
             self._arrive_at_edge_end(t, info)
     ```

2. `src/env/graph_env.py`, Lines 602-609
   - **Reason**: Mathematical correctness violation (A* Admissibility). The A* `_heuristic` function hardcodes an EPSG:4326 degree-to-meter conversion (multiplying coordinate differences by `111000.0`). However, in `_normalize_graph_attributes`, if edge distances are missing, it defaults to raw Euclidean distance (`hypot(vx - ux, vy - uy)`) without the conversion. If the environment is initialized with abstract Euclidean coordinates (like the default `x=1.0, y=0.0`), the actual edge distance is `1.0`, but the heuristic evaluates to `111000.0`. This massive overestimation violates the fundamental A* admissibility requirement ($h(x) \le d(x)$), causing `nx.astar_path` to degrade into greedy best-first search and return mathematically incorrect, suboptimal routes.
   - **Suggested Fix**: Remove the hardcoded lat/lon conversion to ensure the heuristic strictly bounds the un-congested distance.
     ```python
     def _heuristic(self, u: NodeId, v: NodeId) -> float:
         ux, uy = self._node_coords[u]
         vx, vy = self._node_coords[v]
         return hypot(vx - ux, vy - uy)
     ```

3. `src/env/graph_env.py`, Lines 526 and 414
   - **Reason**: Breach of the "Perfect Determinism" dogma. `_apply_dispatch` and `_apply_next_hop_dispatch` iterate directly over `dispatch_actions.items()` and `next_hop_dispatch.items()`. Because Python dictionaries preserve insertion order, if an external caller (or RL policy dict) inserts actions in a different order (e.g. `{2: node_A, 1: node_B}` vs `{1: node_B, 2: node_A}`), the loop processes them in that arbitrary order. For paths of length 1 (immediate node stops), demand is served instantly during this loop. Thus, insertion order dictates which truck consumes the demand when multiple trucks dispatch to the same node in the same tick.
   - **Suggested Fix**: Sort the dictionary items by `truck_id` before iterating to guarantee identical resolution order regardless of construction.
     ```python
     # In _apply_dispatch (Line 526):
     for truck_id, destination in sorted(dispatch_actions.items()):
     
     # In _apply_next_hop_dispatch (Line 414):
     for truck_id, neighbor in sorted(next_hop_dispatch.items()):
     ```


## Physics & Performance

### 1. Object Churn in Antagonist Mask Generation
*   **File & Line:** `src/env/smdp_wrapper.py`, lines 493-497
*   **Cost:** The `valid_levels` list comprehension filters `self.config.congestion_levels` based solely on `self.budget.remaining`. However, this is placed inside the `for edge in sorted(list(nearby_edges)):` loop, meaning the exact same list is recreated from scratch for every single nearby edge. This causes O(E) unnecessary list creations and memory churn per antagonist step.
*   **Fix:** Hoist the `valid_levels` calculation outside the `for edge` loop. Compute it once, and reuse it for all edges in the loop.

### 2. NetworkX Overhead and Slow String Sorting in Hot-Path
*   **File & Line:** `src/env/smdp_wrapper.py`, line 409
*   **Cost:** `sorted(self.env.graph.neighbors(truck.current_node), key=repr)` is called for every idle truck in `_next_hop_action_mask`. `self.env.graph.neighbors` incurs NetworkX dictionary view/generator overhead, and `key=repr` invokes slow Python string formatting `repr()` for every `NodeId` comparison. This is extremely slow to run inside the per-tick hot path.
*   **Fix:** Precompute and cache the sorted neighbors for each node once during environment initialization (e.g., `self.env._sorted_neighbors[node]`), providing O(1) list retrieval.

### 3. O(N) Path Scanning for Dynamic Rerouting
*   **File & Line:** `src/env/smdp_wrapper.py`, lines 246-252
*   **Cost:** In `step_antagonist`, when an edge is congested, the code loops over the remainder of `truck.path` (`for i in range(k, len(truck.path) - 1):`) and calls `self.env._edge_key(u, v)` on every segment to find if the congested edge is in the route. This is an unvectorized O(PathLength) Python loop running inside a nested loop for every truck, exacerbated by the tuple/sorting overhead of `_edge_key`.
*   **Fix:** Equip the truck with an O(1) lookup mapping from edge to path index (e.g., `truck.edge_to_path_index_map`), or at minimum cache `_edge_key` lookups.

### 4. Graph Algorithm Cache Invalidation (Dijkstra)
*   **File & Line:** `src/env/smdp_wrapper.py`, line 180 (and lines 457-461)
*   **Cost:** `self._goal_dist_cache = {}` clears the cached `nx.single_source_dijkstra_path_length` results at every `reset_decision_env()`. Since the base physical graph topology does not change across episodes, dumping and recomputing Dijkstra shortest paths for every customer in every episode is a massive, unnecessary O(V log V + E) overhead.
*   **Fix:** Persist `_goal_dist_cache` across episodes at the class level or inside `GraphEnv`, clearing it only if the underlying map topology fundamentally changes.

### 5. Unvectorized Demand Processing Loop
*   **File & Line:** `src/env/smdp_wrapper.py`, lines 373-380
*   **Cost:** In `protagonist_action_mask`, for every idle truck, an inner loop iterates over all valid customers (`for n, node_demand in valid_customers_by_comp.get(...).items():`) to subtract `other_targeted` loads and append to `destinations`. This is an O(Trucks * Customers) pure-Python `while/for` loop executing heavily in the hot-path.
*   **Fix:** Vectorize the remaining demand calculation using NumPy. Represent demands and targeted loads as arrays, so `unassigned_demand = node_demand_array - targeted_load_array` operates in C, or maintain an O(1) incremental counter of `unassigned_demand` for each customer when truck destinations are set.

## Physics & Performance

### 1. Unvectorized Python Loops & Object Churn in Hot-Path
* **Location:** `src/env/graph_env.py`, lines 258-303 (specifically lines 267-276).
* **Cost:** The movement physics `while True` loop allocates new NumPy arrays (`eff_speeds`, `rem_dists`) on every sub-tick. It also uses a pure Python `for` loop to look up edge properties in NetworkX (`self.graph.edges[u, v]`) for every active truck. NetworkX dictionary lookups and array allocations inside a sub-step loop create a massive bottleneck.
* **Fix:** Maintain a persistent, vectorized state of truck edge-progress and edge-properties (NumPy arrays). Update truck physics using 1D vector operations across all trucks simultaneously, completely avoiding Python `for` loops and NetworkX graph lookups in `step()`.

### 2. Aggressive A* Cache Invalidation
* **Location:** `src/env/graph_env.py`, lines 246-247.
* **Cost:** `self._get_shortest_path.cache_clear()` blows away the entire `lru_cache` for A* routing whenever *any* edge congestion is updated. If congestion updates dynamically, this degrades routing to O(V log V + E) for every dispatch.
* **Fix:** Only invalidate cached paths that actually traverse the congested edge, or replace `nx.astar_path` with a compiled C++/NumPy routing backend (like `scipy.sparse.csgraph.shortest_path`) for rapid shortest-path recalculation.

### 3. Memory Allocation Bottleneck (Object Churn)
* **Location:** `src/env/graph_env.py`, lines 317-330 (inside `observe()`).
* **Cost:** On every single tick, `observe()` dynamically allocates a new nested Python dictionary to represent the `trucks` state. This creates severe garbage collection churn and memory fragmentation during RL sampling.
* **Fix:** Pre-allocate a fixed-size NumPy array (or contiguous buffer) for truck observations. Update it in-place during `step()` and return a memory view instead of constantly instantiating new Python dictionary objects.

### 4. Trigonometry inside A* Heuristic Hot-Path
* **Location:** `src/env/graph_env.py`, lines 602-610 (`_heuristic`).
* **Cost:** The A* heuristic executes `import math` and performs trigonometric calculations (`math.cos`, `math.radians`) inside the inner node-expansion loop of the pathfinder. This is an immense performance sink because it runs for every single node evaluated by A*.
* **Fix:** Pre-project all node coordinates into an orthogonal Cartesian space (e.g., local meters) once during `__init__`. The heuristic can then be reduced to a fast O(1) `hypot(dx, dy)` lookup without runtime trigonometry.

### 5. Step Info Dictionary Allocation
* **Location:** `src/env/graph_env.py`, lines 234-243.
* **Cost:** The `step()` function allocates a large dictionary containing multiple empty Python lists (`dispatched`, `ignored_dispatches`, `congestion_updates`, etc.) every tick, even when nothing happens.
* **Fix:** Cache and reuse a static `info` dictionary structure, clearing the lists via `.clear()` instead of instantiating new lists every tick.
