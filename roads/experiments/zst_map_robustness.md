# zst_map_robustness: threat-map dependence and intel-error robustness

Registered 2026-07-12. Results 2026-07-12. Eval-only, no training.

Artefacts: `models/runs/zst_map_robustness.json`, policy checkpoint `models/runs/gen16_multicity/seed0_ckpts/actor_ep1000.pt`. Scripts: `analysis/map_robustness_eval.py`, `analysis/zst_kn_rows.py`.

## Question

Does the frozen multi-city generalist's zero-shot edge depend on reading the threat map, and does it survive corrupted threat observations?

## Game

- Frozen policy: the gen16 seed-0 actor, TAP over the three checkpoints centred on its selected best (ep 500/1000/1500), the estimator used for the gen16 zero-shot K/N rows.
- Instances: the 6 held-out Gdansk ODs (pool-seed 0, the gen16 test set).
- A random-init reference network is evaluated on identical footing in every condition.
- A2, shuffled reality: per instance, 3 seeded permutations of the vulnerability values across its candidate edges, non-candidate edges keeping true values. Each shuffled map defines a new game with the same routes, whose equilibrium, loss_det and best-response matrix are recomputed; the policy observes the shuffled map, including recomputed per-route [cost, worst-vuln] features, and is scored under the shuffled game's oracle best response, as a ratio to the shuffled game's equilibrium.
- A3, intel error: the true game scores everything, only the observed map is corrupted. Shuffle-fraction f in {0.25, 0.5, 1.0} over a seeded random subset of candidate edges, and multiplicative noise sigma in {0.1, 0.25, 0.5} with p' = clip(p*(1+eps), 0.05, 0.99), eps ~ N(0, sigma); 3 draws each; route features recomputed from the corrupted map.
- Context measured before the run: route cost against route worst-vulnerability correlation |corr| 0.60-0.99 on 8/8 pool instances, and geometry-decorrelated maps move the equilibrium strategy by L1 0.44-1.03.

```bash
PYTHONPATH=. .venv/bin/python analysis/map_robustness_eval.py \
  models/runs/gen16_multicity/seed0_ckpts/actor_ep1000.pt \
  --json-out models/runs/zst_map_robustness.json
```

## Criteria

- A2 tracks the map: shuffled-map mean ratio <= 2.0 and below the random-init reference on the same shuffled games. At or above random-init means the policy reads geometry rather than the map; in between is partial, reported as measured.
- A3 graceful: the mean ratio degrades monotonically but stays below random-init up to f = 0.5 and sigma = 0.25. A fall to or past random-init at the smallest corruption is a cliff.
- Sanity row, required either way: the true-map ratio through this harness must reproduce ~1.7x, the gen16 N3K1 value.

## Baselines

- random-init: an untrained network in every condition, on identical footing.
- loss_det: each game's deterministic-class optimum.
- True-map policy row: the frozen policy with no corruption.
- Constant-map diagnostic (post-hoc, labelled as such): all candidate vulnerabilities observed as 0.55.

## Results

| condition | gen (TAP, seed-0 window 500/1000/1500) | random-init | note |
|---|---|---|---|
| Sanity: true map, true game | 1.71x [2.13, 2.33, 1.20, 1.69, 1.52, 1.41] | 1.99x | reproduces the gen16 N3K1 sanity row |
| A2: shuffled reality, 18 cells | 1.80x | 2.19x | beats loss_det 13/18 |
| A3 shuffle-fraction 0.25 / 0.5 / 1.0 | 1.70 / 1.72 / 1.74 | 1.99 | reality true, observation corrupted |
| A3 multiplicative sigma 0.1 / 0.25 / 0.5 | 1.72 / 1.78 / 1.78 | 1.99 | reality true, observation corrupted |
| Diagnostic: information-free constant map | 1.80x | 1.99 | all candidate vulns observed as 0.55 |

A2 criterion met (1.80x <= 2.0 and below random-init 2.19x). A3 criterion met at every level tested, a fully shuffled observed map costing +0.03 (1.71 to 1.74) and the strongest multiplicative noise costing +0.07. The constant-map diagnostic costs +0.09 (1.71 to 1.80), still below random-init 1.99.
