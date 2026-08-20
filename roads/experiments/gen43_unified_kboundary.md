# gen43: interdiction-budget ladder, one instance, both registers

Registered 2026-08-08. Results 2026-08-08 to 2026-08-10.
Code 261703c (static batch, dynamic K=1/4), c59aa9d (dynamic K=5/6), 9630cf8 and 4bd8f02
(exact benchmark extensions). Reused cells trained at 152f880 (static K=5), 8ba949e
(static K=6), 5af4dd1 (dynamic K=2/3 and the no-window control).
Artefacts: models/runs/gen43_unified/; analysis/gen43_consolidation_probe.py,
analysis/gen43_dyn_highk_probe.py, analysis/gen43_static_exact_highk.py,
analysis/gen43_static_exact_k78.py, analysis/gen43_reported_rows.py, analysis/dyn_exact.py,
analysis/gen43_batch.sh, analysis/gen43_dyn_ext_batch.sh.

## Question

How does trained performance move with the interdiction budget K in the static and dynamic
registers on one instance, against exact benchmarks and the full rule family.

## Game

- Königsberg road graph, OD 71-33. Six edge-disjoint corridors, menu of 11 routes, fleet
  N=3, interception band (0.15, 0.95), mission objective (probability at least one convoy
  is lost).
- Static register. The adversary commits K edges and best-responds to the defender's
  strategy. Trainer scripts/train_multiconvoy.py, smooth fictitious play, 1,200 sorties,
  3 seeds, best-checkpoint selection.
- Dynamic register. The adversary softmax-responds (tau 0.15) to the trailing 3-episode
  route window. Trainer scripts/train_b1lite1.py, 8,000 sorties, 3 seeds, best-checkpoint
  selection.
- Benchmarks. Static game value v* by exact LP (stacked form; constraint generation at
  K>=7, certificate gaps at most 2.6e-08). Dynamic optimum by Karp minimum mean cycle
  (analysis/dyn_exact.py). Trained static cells at K>=5 scored under the certified greedy
  best response, whose deviation from the exact response measured 0.0000 on every stack
  arm at every budget where both were computed.

## Criteria

- Dynamic K>=3 cells. Best-checkpoint stationary per-sortie loss below the best rule on
  at least 2/3 seeds and pooled. Strong bar, pooled at most 1.15x the exact optimum.
- Static cells. Descriptive, no superiority bar.
- Reuse seam. The new K=4 and K=7 cells must bracket the reused K=5/6 cells within 3
  pooled standard deviations of trend.

## Baselines

- Uniform stack over the six disjoint corridors.
- Inverse-vulnerability stack, worst-edge weighting (mass proportional to
  1/(1-(1-max_e p_e)^N), fixed across K).
- Inverse-vulnerability stack, budget-max weighting (mass from the worst K-edge attack on
  each route).
- Both stacks over the full 11-route menu.
- Best committed route (deterministic value).
- Tabular smooth fictitious play with the same best-response oracle.
- Dynamic rules. Best rotation over 20 corridor orders, composed anti-repeat, full-menu
  anti-repeat, iid equilibrium mixture, matched-budget tabular window-Q.

## Results

Static register (time-average interception, 3 seeds):

| K | v* exact | best stack | tabular FP | SACRED (seeds) | pooled |
|---|---|---|---|---|---|
| 1 | 0.127640 | 0.127640 | 0.127640 | 0.163 / 0.162 / 0.156 | 0.160 +/- 0.003 |
| 2 | 0.255280 | 0.255280 | 0.255280 | 0.325 / 0.335 / 0.324 | 0.328 +/- 0.005 |
| 3 | 0.382920 | 0.382920 | 0.382920 | 0.462 / 0.471 / 0.455 | 0.463 +/- 0.007 |
| 4 | 0.510560 | 0.510560 | 0.510560 | 0.579 / 0.614 / 0.622 | 0.605 +/- 0.018 |
| 5 | 0.620058 | 0.638200 | 0.621 | reused | 0.667 +/- 0.016 |
| 6 | 0.686494 | 0.7298 | 0.690 | reused | 0.733 +/- 0.015 |
| 7 | 0.752166 | 0.7844 | 0.759 | 0.771 / 0.772 / 0.792 | 0.778 +/- 0.010 |
| 8 | 0.806521 | 0.8216 | 0.812 | 0.820 / 0.819 / 0.826 | 0.822 +/- 0.003 |
| 9, 10 | 0.832529 | det 0.832529 | not run | not trained | - |

- Best stack per K. Inverse-vulnerability disjoint (worst-edge) at K=1-5,
  inverse-vulnerability full-menu at K=6/7, uniform full-menu at K=8.
- The worst-edge disjoint stack equals v* at K=1, 2, 3 and 4 (gap 0.00e+00); first
  exceeded at K=5 (0.638200 vs 0.620058). Both inverse-vulnerability weightings, K=1-6:
  worst-edge 0.127640 / 0.255280 / 0.382920 / 0.510560 / 0.638200 / 0.765839, budget-max
  0.127640 / 0.297795 / 0.455620 / 0.585971 / 0.701898 / 0.794721.
- Best committed route 0.832529 from K=3 upward. Value of mixing over it, 25.52% (K=5),
  17.54% (K=6), 9.65% (K=7), 3.12% (K=8), 0.0000% (K=9). v*(9) = v*(10) = 0.832529.
- SACRED meets or beats the best stack at no static budget. Reuse-seam criterion met, the
  trained curve is smooth across the reused K=5/6 cells.

Dynamic register (stationary per-sortie loss, 3 seeds, all references exact):

| K | optimum | best rule | window-Q | iid_eq | SACRED (seeds) | pooled | criterion |
|---|---|---|---|---|---|---|---|
| 1 | 0.0313 | 0.0387 | 0.0472 | 0.0967 | 0.0467 / 0.0468 / 0.0450 | 0.0462 +/- 0.0008 | rule ahead |
| 2 | 0.0657 | 0.0929 | 0.1083 | 0.1823 | reused | 0.0934 | tie |
| 3 | 0.1018 | 0.1539 | 0.1759 | 0.2549 | reused | 0.1406 | met 3/3 (-8.6%) |
| 4 | 0.1386 | 0.2152 | 0.2169 | 0.3117 | 0.1774 / 0.1823 / 0.1863 | 0.1820 +/- 0.0036 | met 3/3 (-15.4%) |
| 5 | 0.1756 | 0.2743 | 0.2535 | 0.3593 | 0.2151 / 0.2233 / 0.2141 | 0.2175 +/- 0.0041 | met 3/3 (-20.7%) |
| 6 | 0.2121 | 0.3295 | 0.3159 | 0.4024 | 0.2612 / 0.2659 / 0.2642 | 0.2638 +/- 0.0020 | met 3/3 (-19.9%) |

- Strong bar unmet at every K. Pooled ratio to the exact optimum 1.38x (K=3), 1.313x
  (K=4), 1.239x (K=5), 1.244x (K=6).
- No-window control at K=3, 0.2328.
- Slack over the best rule collected by SACRED, 26% (K=3), 43% (K=4), 57.5% (K=5),
  56.0% (K=6).
- Committed-adversary exploitability of the best seed's marginal mixture, as a ratio of
  the one-shot equilibrium value, 1.60 / 1.72 / 1.51 / 1.35 / 1.24 / 1.17 at K=1-6.
- K=7 and K=8 not trained. The exact loss matrix is 2.8 GB per process at K=7 and 12.4 GB
  at K=8.
