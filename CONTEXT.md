# CONTEXT.md (The Blueprint)

## 1. Current Architecture
The **SACRED** framework (Soft Actor-Critic Robust Evolutionary Deep reinforcement learning) models the Stochastic and Dynamic Vehicle Routing Problem (SDVRP) as an **Asymmetric Zero-Sum Markov Game**.

*   **Protagonist (Central Dispatcher):** Aims to minimize delivery times and fulfill customer demands using a fleet of trucks.
*   **Antagonist (Adversary):** Aims to maximize disruption by dynamically injecting traffic congestion, constrained by a time/cost budget.

### Module Map
*   `src/env/graph_env.py`: The O(1) tick-by-tick physics and routing engine. Holds core determinism.
*   `src/env/smdp_wrapper.py`: The event-driven wrapper collapsing ticks into `DecisionType.PROTAGONIST_DECISION` and `DecisionType.ANTAGONIST_DECISION`.
*   `src/agents/sacred_atla.py`: The trainer executing Alternating Training with Learned Adversaries (ATLA). Alternates between protagonist and antagonist epochs, freezing weights of the inactive agent.

## 2. Active Epic
**Phase 2: Master Thesis Experimentation.**
Phase 1 Engineering is formally complete. The Master Audit successfully triaged and eliminated all critical `O(N)` bottlenecks, deepcopy memory leaks, and dimensionality crashes across the `src/env/`, `src/agents/`, `src/sbo/`, and `src/baselines/` domains. Furthermore, we mathematically cured SAC Entropy Collapse (Premature Convergence) by introducing reward scaling, lowered learning rates, and strict Target Entropy constraints. The ATLA phase-switching logic is verified to produce robust co-evolutionary Q-value staircases.
We are now focused on executing the massive 1000-2000 episode thesis runs and recording PyGame visualization metrics via `scripts/evaluate_agents.py`.

## 3. Known Debt / Quirks
*   **SAC Entropy Dynamics:** To prevent deterministic overfitting on complex graphs, we scale environment rewards by `0.001` to stabilize gradients and explicitly enforce a `target_entropy=-1.0` constraint. `lr_actor` is intentionally bottlenecked at `5e-5` to favor stable co-evolution over fast, brittle convergence.
*   **PyG Tensor Caching:** `edge_index_tensor` is permanently cached in `_FEATURIZE_CACHE` during `featurize_state()`, converting PyTorch topology allocations to an $O(E)$ dynamic list comprehension.
*   **Topological Crash Protection:** Graph connected components are precomputed. Protagonist action masking inherently skips isolated geometries to avoid crashing the A* algorithm.
*   **Antagonist Action Space:** We explicitly reject `MultiDiscrete` to avoid hierarchical conditional masking. The Antagonist relies on a flattened `Discrete(E * L + 1)` space.
