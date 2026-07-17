# The regime decision table: which routing policy to deploy, by adversary type x budget

- **status: SYNTHESIS (2026-07-17, eval/oracle-only; the practical artefact the thesis's
  application chapter turns on). Numbers cite the ledgers, not recomputed here.**

The whole programme, reduced to the decision a logistics planner actually faces. Two axes:
the adversary's INFORMATION (does it best-respond to your STRATEGY, or only react to your recent
PATTERN?) and the interdiction BUDGET relative to the min-cut m. Cell entries: the recommended
policy and the measured evidence.

| | **Low budget (K << m)** | **High budget (K -> m), or the enumeration wall (K >= 4)** |
|---|---|---|
| **Strategy-aware adversary** (best-responds to your mixed strategy; the minimax register) | **Use the 2-line max-flow heuristic.** Uniform-stack over the edge-disjoint routes is within ~1.2x the equilibrium and matches trained SACRED (35-159 K=1: heuristic 0.250 ~ SACRED 0.256; population 4/4 ODs <=1.25x eq). Learning buys only a ~10% fleet-cost saving (R0b) and the discovery that it FOUND the structure unaided. | **Use adversarially self-played SACRED.** At K=m-1 it beats both heuristic variants on the exact yardstick (35-159 K=3: 0.664 vs 0.738); past the wall (71-33 m=6, greedy yardstick) it beats uniform-disjoint at K=5 (0.667 vs 0.705). Caveat (gen26 second pass): a tabular-FP learner with the same greedy BR also works here, so the claim is "learning is REQUIRED, deep RL is one sufficient method", carried by the boundary map. |
| **Pattern-of-life adversary** (reacts to your recent realised routes; the quantal-response register) | **Use a history-aware policy (gen27) OR the composed disjoint+anti-repeat rule.** Both beat every static object; the composed 2-line rule (0.50-0.61x the static cap) slightly leads the trained policy (0.639), which MATCHES it zero-shot on unseen cities having discovered both insights unaided (window-avoidance -0.29, disjoint-core mass 0.50). | **Use a history-aware policy.** The dynamic advantage and the high-budget advantage compound; no static rule (heuristic or LP) can follow either axis. (Aerial act, gen28, extends this to continuous coverage.) |

**The honest through-line (binding for the application chapter):** *contested routing needs no
learning in the low-budget strategy-aware corner — a two-line heuristic is near-optimal and the
thesis proves it. Learning becomes necessary as the interdiction budget approaches the min-cut
(where exact solvers also fail) and against pattern-of-life adversaries (where every static method
is capped). SACRED is a sufficient method in both regimes; the contribution is the MAP of where
each tool is the right one, scored against computable optima throughout.*

**LLM row (B2, 35-159, both open-weight models):** language agents unaided sit in NEITHER
recommended cell — reg-(b) 0.52-0.60, worse than the heuristic and naive stacking, despite
understanding the structure; in-context feedback (reg-c) partially recovers via emergent
anti-repeat but stays short of the trained policy. Calibrated randomisation is the capability
they lack unaided.
