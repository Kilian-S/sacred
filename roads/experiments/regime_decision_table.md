# regime_decision_table: recommended routing policy by adversary model and budget

Synthesis of ledgered results. Every number cites its ledger; m is the instance's number
of edge-disjoint corridors.

| adversary | low budget (K << m) | high budget (K -> m) |
|---|---|---|
| Strategy-aware (best-responds to the defender's strategy) | Inverse-vulnerability disjoint stack. Equals the exact game value at K=1-4 on the six-corridor instance (gen43). | Naive stacks or tabular fictitious play. Trained SACRED never beats the best stack (gen43); mixing's value over the best committed route reaches zero at K=9. |
| Observant (responds to the defender's recent realised routes) | Rotation ahead at K=1, tie at K=2 (gen43). | SACRED. Beats every rule at K=3-6, margins 8.6% to 20.7% (gen43); transfers zero-shot to unseen cities at K=1 (gen27). |
| Doctrine described in natural language | Type-conditioned SACRED with LLM doctrine identification (gen38). | Same. |

Unaided language models fit no cell. Committed mixtures score worse than naive stacking
or worse than the best single route (b2); in-context play stays behind the trained policy
on every instance (b2).
