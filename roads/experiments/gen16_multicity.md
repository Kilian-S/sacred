# gen16: multi-city generalist, cross-city zero-shot transfer

Registered 2026-07-10. Results 2026-07-11 (three seeds, K/N rows, Kyiv row), 2026-07-16 (disjoint-baseline rows).

Artefacts: `models/runs/gen16_multicity/seed{0,1,2}_ckpts` (per-eval checkpoints; the seed-0 selected actor is `models/runs/gen16_multicity/seed0_ckpts/actor_ep1000.pt`), `models/runs/r0_screen.json`. Scripts: `scripts/train_generalist.py`, `analysis/repair_map_lengths.py`, `analysis/zst_kn_rows.py`, `analysis/disjoint_baseline_probe.py`, `analysis/r0_screen.py`.

## Question

Does one policy trained on three cities route fleets zero-shot on a fourth city held out entirely?

## Game

Cities, all built by the same arterial-filter and 30m-consolidation pipeline, oracle-screened on 8 sampled instances.

| city | nodes | role | screen |
|---|---|---|---|
| Kaliningrad (30m) | 290 | train | the campaign graph |
| East London | 564 | train | eq med 0.264, det/eq 2.29 |
| Istanbul | 1266 | train | eq med 0.263, det/eq 2.16 |
| Gdansk | 356 | held out, zero-shot | eq med 0.307, det/eq 2.41 |

- Config: `scripts/train_generalist.py --cities kaliningrad,east_london,istanbul --holdout-city gdansk --n-per-city 6 --n-test 6`, giving 18 train instances and 6 held-out-city test instances; pool-seed 0 fixed across seeds.
- Otherwise the gen15 recipe verbatim: per-instance smooth fictitious play, transferable head features only at lr 3e-2, edge-vulnerability observation, per-transition menus, fleet-route, role alphas, 12,000 sorties, eval-every 500, exact per-instance evaluation, per-eval checkpoints.
- Seeds {0,1,2}, `--threads 3`, three runs in parallel.
- Selection: best checkpoint, reported under both select-on-test (the held-out-city mean) and select-on-train.
- Map lengths were repaired from geometry before the pool build (`analysis/repair_map_lengths.py`).

## Criteria

Primary metric: the held-out city's 6-OD mean best-checkpoint TAP ratio, each OD scored against its own oracle equilibrium, pooled over 3 seeds.

- PASS: pooled mean <= 2.0, below the random-init reference on the same ODs, and beating each OD's loss_det on >= 4/6 ODs.
- STRONG: <= 1.7 and beating loss_det on 6/6.
- Secondaries: per-train-city ratios, the A2-rescue row on kaliningrad_original, the `route_feat_w` trajectory, and the train-to-held-out generalisation gap.

## Baselines

- random-init: an untrained network evaluated on identical footing.
- loss_det: the OD's deterministic-class optimum.
- uniform-disjoint stack: uniform stack over the edge-disjoint routes, no training and no labels.
- inverse-vuln variant: the inverse-vulnerability weighted stack over the same routes.
- gen15 single-city in-graph transfer, 1.59; single-source cross-graph transfer, at random-init level.

## Results

| seed | best-checkpoint held-out-city mean ratio @ sortie | per-Gdansk-OD | train ratio there |
|---|---|---|---|
| 0 | 1.599 @ 1000 | 1.56 / 1.89 / 1.36 / 1.60 / 1.58 / 1.60 | 1.54 |
| 1 | 1.773 @ 500 | 1.28 / 1.99 / 2.42 / 1.38 / 1.59 / 1.98 | 1.69 |
| 2 | 1.660 @ 500 | 1.39 / 2.30 / 1.49 / 1.32 / 2.03 / 1.42 | 1.75 |

Held-out-city (Gdansk) best-checkpoint mean ratio 1.677 +/- 0.072 over 3 seeds. PASS criterion met on every clause: pooled mean 1.677 <= 2.0; below the random-init reference (1.68 against random ~1.99 on the same ODs); loss_det beaten on 17/18 (OD, seed) cells, per seed 6/6, 5/6, 6/6. STRONG criterion not met, the pooled mean meets 1.7 (1.677) but one cell misses by 0.01 (seed 1's OD 193-278 at 1.98x against that OD's loss_det at 1.97x).

Selection dual-report: select-on-test 1.677 +/- 0.072 against select-on-train 1.733 +/- 0.149 (seed 1 moves 1.773 to 1.941); final iterate 2.20. Both clear the bar and sit below the ~1.99 random-init reference.

A2-rescue row: on kaliningrad_original, where the single-source actor tied random-init (2.40 against 2.41), the multi-city actor scores 1.90 against random 2.43, zero-shot on a graph it never trained on.

Zero-shot K/N rows (2026-07-11, eval-only): the frozen seed-0 best-checkpoint actor, trained at N=3, K=1, evaluated without retraining on the held-out Gdansk ODs at shifted adversary budget and fleet size, scored against each (OD, K, N) cell's own oracle equilibrium.

| cell | gen (TAP) | random-init | beats loss_det |
|---|---|---|---|
| N=3 K=1 (train regime, sanity) | 1.71x | 1.99x | 6/6 |
| N=3 K=2 (budget shift) | 1.29x | 1.34x | 5/6 |
| N=5 K=1 (fleet shift) | 1.79x | 2.10x | 6/6 |

Scale-axis row (2026-07-11, eval-only): the same frozen actor on the whole Kyiv arterial network (6083 nodes, 10861 edges), 5 screened held-out ODs, single-checkpoint exact evaluation for both arms.

| | mean ratio to eq | beats loss_det | per-OD |
|---|---|---|---|
| generalist | 1.88x | 3/5 | 1.33 / 1.77 / 1.63 / 2.68 / 1.97 |
| random-init | 2.03x | - | 2.02 / 2.18 / 2.09 / 2.41 / 1.47 |

That row is a partial pass, with mean 1.88x <= 2.0, below random-init 2.03x, and loss_det beaten on 3/5 ODs. Kyiv screens as less asymmetric than the training cities (eq med 0.253, loss_det/eq med 1.96, against the cities' 2.2-2.4).

Disjoint-baseline rows (2026-07-16, oracle and eval-only) on the same 6 held-out Gdansk ODs: uniform-disjoint stack 1.134x eq, beating loss_det 6/6; inverse-vuln variant 1.024x eq. Neither uses training, labels, graph exposure or a threat map. The transfer ladder on these ODs reads distill 1.555 < retrieval 1.676 < adversarial 1.733 < random ~1.99 < vanilla 2.354, with the heuristic at 1.134 below all of them. Structure row: zero-shot, the policy places 0.54-0.89 of its mass on each instance's disjoint core, where the equilibrium allocates 0.53-0.97 and uniform allocates ~0.28.

Scope of the run: 6 ODs per held-out graph, N=3, K=1, best-checkpoint selection.
