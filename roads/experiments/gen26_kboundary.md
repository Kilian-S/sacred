# gen26: static cells at budgets near the corridor count

Registered 2026-07-16. Results 2026-07-16 to 2026-07-17.
Code c9c474a (35-159 K=3), 152f880 (71-33 K=5), 8ba949e (71-33 K=6).
Artefacts: models/runs/gen26_kboundary/; analysis/disjoint_baseline_probe.py.

## Question

Does trained play beat the strongest naive stacks as the interdiction budget K approaches
the number of edge-disjoint corridors m in the static register.

## Game

- Königsberg road graph, fleet N=3, interception band (0.15, 0.95), mission objective.
  Two instances: OD 35-159 (m=4, K=3, exact attacker) and OD 71-33 (m=6, menu of 11
  routes, K=5 and 6, certified greedy best response; measured deviation from the exact
  response at most 1.8% at K<=3 on this instance).
- The adversary commits K edges and best-responds to the defender's strategy. Trainer
  scripts/train_multiconvoy.py, smooth fictitious play, 1,200 sorties, 3 seeds,
  best-checkpoint selection.

## Criteria

- 35-159 K=3: best-checkpoint value below the uniform disjoint stack (0.738) on at least
  2/3 seeds and pooled. Strong bar, pooled at most 0.68.
- 71-33 K=5: below the uniform disjoint stack (0.705) on at least 2/3 seeds and pooled.
  Strong bar, below the inverse-vulnerability disjoint stack (0.638).
- 71-33 K=6, n=3 gate: pooled below the best naive stack of any class (0.739).

## Baselines

- Uniform and inverse-vulnerability stacks over the disjoint corridors.
- Uniform and inverse-vulnerability stacks over the full menu.
- Best committed route; exact equilibrium at K<=3.
- Tabular smooth fictitious play with the same best-response oracle.

## Results

| instance | K | best stack | tabular FP | SACRED (seeds) | pooled | criterion |
|---|---|---|---|---|---|---|
| 35-159 | 3 | 0.737 | - | 0.656 / 0.647 / 0.690 | 0.664 +/- 0.018 | met 3/3; strong met |
| 71-33 | 5 | 0.638 | 0.621 | 0.690 / 0.656 / 0.654 | 0.667 +/- 0.016 | met against uniform 0.705 on 3/3; strong unmet |
| 71-33 | 6 | 0.730 | 0.690 | 0.718 / 0.728 / 0.754 | 0.733 +/- 0.015 | unmet; pooled 0.733 against the 0.739 bar, 2/3 seeds below |

- 35-159 K=3 equilibrium 0.604; best committed route 0.933. 71-33 equilibria beyond the
  exact wall at this act's code state; exact values recorded later in gen43.
- 71-33 stack values under the greedy yardstick: K=5 uniform disjoint 0.705,
  inverse-vulnerability disjoint 0.638, uniform full menu 0.666, inverse-vulnerability
  full menu 0.667; K=6 0.800 / 0.766 / 0.739 / 0.730.
- Final iterates drift above the best checkpoints at K=3 (0.795 / 0.954 / 0.833); at K=5
  the finals sit within 0.02 of the best checkpoints (0.708 / 0.656 / 0.659).
