append_text = """

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
"""

with open('MASTER_AUDIT.md', 'a') as f:
    f.write(append_text)
