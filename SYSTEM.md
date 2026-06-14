# SYSTEM.md (The Identity)

## 1. Behavioral Guidelines
These behavioral rules exist to reduce common LLM coding mistakes. Bias toward caution over speed.

*   **Think Before Coding:** Don't assume. Surface tradeoffs. If uncertain, explicitly state your assumptions and ask for clarification. If multiple interpretations exist, present them.
*   **Simplicity First:** Minimum code that solves the problem. No speculative features, no abstractions for single-use code, no "flexibility" that wasn't requested. If you write 200 lines and it could be 50, rewrite it.
*   **Surgical Changes:** Touch only what you must. Clean up only your own mess. Don't refactor things that aren't broken, and perfectly match the existing style.

## 2. Tech Stack & Hardware
*   **Hardware:** Apple Silicon (M4 Mac with 24GB RAM).
*   **Language:** Python 3.10+
*   **Core Libraries:** `NetworkX` (Graph Math), `PyTorch` + `PyTorch Geometric (PyG)` (Deep Learning), `PyGame` (Visualization).
*   **Environment API:** Custom SMDP Event-Wrapper over a headless simulator, transitioning to PettingZoo/Gym.
*   **Algorithm Base:** Modified Soft Actor-Critic (SAC).

## 3. Strict Design Patterns & Dogma
*   **Perfect Determinism:** All operations must yield mathematically identical results across runs. When processing unordered sets or dicts for heuristic baselines, always use `sorted(list(...))` before executing distance calculations or state updates.
*   **O(1) Computations:** The simulation relies on heavy caching (`functools.lru_cache`, `_FEATURIZE_CACHE`) and fast native set intersections. Never introduce deep nested loops or unnecessary object allocations (`copy.deepcopy()`) inside the hot-paths like `observe()` or `step()`.
*   **Separation of Concerns:** The physics engine (`graph_env.py`) must remain completely unaware of RL hyperparameters. Apply RL-specific logic (like $\gamma$ discounting) purely on the agent wrapper side using `elapsed_ticks`.
*   **Crash-Proof Topology:** Maintain strict logical checks against physics-engine exceptions. The protagonist action mask absolutely must filter out physically unreachable nodes to completely prevent `nx.NetworkXNoPath`.
