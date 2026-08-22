# gen35: dynamic cells at interdiction budgets two and three

Registered 2026-07-23. Results 2026-07-23. Code 5af4dd1.
Artefacts: models/runs/gen35_dyn_kboundary/; analysis/dyn_exact.py,
analysis/gen35_reported_rows.py.

## Question

Does trained play beat every two-line rule in the dynamic register at interdiction
budgets K=2 and K=3 on the six-corridor instance.

## Game

- Königsberg road graph, OD 71-33. Six edge-disjoint corridors, menu of 11 routes, fleet
  N=3, interception band (0.15, 0.95), mission objective.
- The adversary softmax-responds (tau 0.15) to the trailing 3-episode route window.
  Trainer scripts/train_b1lite1.py, 8,000 sorties, 3 seeds, best-checkpoint selection.
- Benchmarks: exact dynamic optimum by Karp minimum mean cycle (analysis/dyn_exact.py);
  iid equilibrium cap by exact enumeration.

## Criteria

- Per cell: best-checkpoint stationary per-sortie loss below the best rule (0.0929 at
  K=2, 0.1539 at K=3) on at least 2/3 seeds and pooled. Strong bar, pooled at most 1.15x
  the exact optimum.
- No-window control at K=3 expected at the memoryless cap.

## Baselines

- Best rotation over 20 corridor orders.
- Composed anti-repeat over the corridors; anti-repeat over the full menu.
- iid equilibrium mixture; best committed route.
- Matched-budget tabular window-Q (the same interaction budget, no network).

## Results

| K | optimum | best rule | window-Q (seeds) | iid_eq | SACRED (seeds) | pooled | criterion |
|---|---|---|---|---|---|---|---|
| 2 | 0.0657 | 0.0929 | 0.1083 (0.0918 / 0.1160 / 0.1172) | 0.1823 | 0.0933 / 0.0919 / 0.0950 | 0.0934 | unmet, 1/3 seeds, pooled +0.5% |
| 3 | 0.1018 | 0.1539 | 0.1759 (0.1596 / 0.1966 / 0.1714) | 0.2549 | 0.1356 / 0.1428 / 0.1435 | 0.1406 | met 3/3 (-8.6%) |

- Strong bar unmet at both cells. Pooled ratio to the exact optimum 1.42x (K=2), 1.38x
  (K=3).
- No-window control at K=3, 0.2328; its window weight trained to 0.00.
- Committed-adversary exploitability of the best seed's marginal mixture, as a ratio of
  the one-shot equilibrium value, 1.72 (K=2), 1.51 (K=3).
