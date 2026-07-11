# Generation: gen16_multicity (the multi-city generalist: cross-CITY zero-shot transfer)

- **status: PRE-REGISTERED 2026-07-10 evening (Kilian: "you can push on the ZST"; maps provided by
  Kilian, length-repaired from geometry after an extractor bug - `scratch/repair_map_lengths.py`);
  binding at launch.**
- **git SHA:** the commit landing this ledger + the multi-city machinery.

## Why (the A2 finding made this the design)

`experiments/a2_graph_transfer.md`: a single-source-graph generalist transfers across OD pairs
(gen15: 1.59x held-out) but NOT across graphs (ties random-init on a different graph) - the GNN
overfits to the one graph it ever saw. The theoretically indicated cure (the review's
catastrophic-overfitting citations: breadth of exposure) is MULTI-GRAPH training. This generation
trains ONE policy across THREE cities and evaluates zero-shot on a FOURTH city held out entirely.

## Cities (all built by the same arterial-filter + 30m-consolidation pipeline; oracle-screened)

| city | nodes | role | screen (8 sampled instances) |
|---|---|---|---|
| Kaliningrad (30m) | 290 | TRAIN | the campaign graph |
| East London | 564 | TRAIN | eq med 0.264, det/eq 2.29 |
| Istanbul | 1266 | TRAIN | eq med 0.263, det/eq 2.16 |
| **Gdansk** | 356 | **HELD OUT (zero-shot)** | eq med 0.307, det/eq 2.41 |

Hold-out choice (recorded): Gdansk is the Baltic-port analogue of Kaliningrad = the operationally
meaningful "unseen comparable theatre" claim; the training set maximises structural variety
(organic Baltic + London arterial + Istanbul mega-city), which is what the A2 mechanism says the
GNN needs.

## Config

`scripts/train_generalist.py --cities kaliningrad,east_london,istanbul --holdout-city gdansk
--n-per-city 6 --n-test 6` (18 train instances, 6 held-out-city test instances; pool-seed 0 fixed
across seeds); otherwise the gen15 recipe verbatim (per-instance smooth FP, transferable head
features only at lr 3e-2, edge-vulnerability observation, per-transition menus, fleet-route, role
alphas, 12,000 sorties, eval-every 500, exact per-instance evaluation, per-eval ckpts), seeds
{0,1,2}, `--threads 3` 3-parallel.

## Decision metric (PRE-REGISTERED)

Primary = **the held-out CITY's 6-OD mean best-checkpoint TAP ratio** (each OD scored against its
own oracle equilibrium; best checkpoint by the held-out mean, the standing discipline), pooled
over 3 seeds.

> **PASS:** pooled mean <= 2.0 AND < the random-init reference on the same ODs AND beats each OD's
> loss_det on >= 4/6 ODs (majority; cross-graph is the hard axis - gen15's every-OD bar was
> near-missed even in-graph). **STRONG:** <= 1.7 AND beats loss_det on 6/6.
> Context: gen15 single-city in-graph = 1.59; single-source cross-graph = ~random (2.40 vs 2.41).
> ANY pass is the first cross-city zero-shot transfer of the programme.

Secondaries: per-train-city ratios (does the policy stay good everywhere it trained?);
**the A2-rescue row** - the multi-city actor re-evaluated on kaliningrad_original (the graph where
the single-source actor tied random): does multi-graph training rescue THAT transfer too?;
`route_feat_w` trajectory; the generalisation gap (train vs held-out).

## RESULT (2026-07-11 ~02:00, 3 seeds, ~4 h): **PASS - the first cross-CITY zero-shot transfer**

| seed | best-ckpt held-out-CITY mean ratio @ sortie | per-Gdansk-OD | train ratio there |
|---|---|---|---|
| 0 | **1.599** @ 1000 | 1.56 / 1.89 / 1.36 / 1.60 / 1.58 / 1.60 | 1.54 |
| 1 | 1.773 @ 500 | 1.28 / 1.99 / 2.42 / 1.38 / 1.59 / 1.98 | 1.69 |
| 2 | 1.660 @ 500 | 1.39 / 2.30 / 1.49 / 1.32 / 2.03 / 1.42 | 1.75 |

> **Held-out-CITY (Gdansk) best-checkpoint mean ratio 1.677 +/- 0.072 (3 seeds).** Gdansk was
> never trained on; every OD is scored against its own oracle equilibrium.

**Against the pre-registered bars: PASS on every clause.**
- pooled mean 1.677 <= 2.0 ✓;
- **< the random-init reference** (1.68 vs random ~1.99 on the same ODs) ✓ - footing note: the
  primary is the TAP (checkpoint-ensemble) read, the project's standing deployable object; the
  single final iterate wobbles to ~2.2 (FP cycling, as every SACRED result), disclosed;
- **beats each OD's loss_det on 17/18 (OD, seed) cells** (seed 1's OD 193-278 misses by 0.01:
  1.98x vs the OD's loss_det at 1.97x) - >= 4/6 required per seed, achieved 6/6, 5/6, 6/6 ✓.
- STRONG (<= 1.7 AND 6/6): the pooled mean meets 1.7 (1.677) but the one 0.01-miss cell denies the
  6/6 clause - narrowly missed, reported as measured.

**THE A2-RESCUE ROW (the mechanism test): CONFIRMED.** On kaliningrad_original - the graph where
the SINGLE-source actor TIED random (2.40 vs 2.41, `experiments/a2_graph_transfer.md`) - the
multi-city actor scores **1.90 vs random 2.43** zero-shot, on a graph it never trained on either.
Multi-graph training is what fixes cross-graph transfer, exactly as the A2 finding predicted: the
GNN needed graph variety, and three cities suffice for a measurable, general effect.

**What is established (the aim-level ZST promise, at the CITY level):** one policy trained on
Kaliningrad + East London + Istanbul routes fleets in never-seen Gdansk at 1.68x its equilibria
zero-shot, beating the deterministic-class optimum on 17/18 cells, and generalises to a second
unseen graph (1.90 vs random 2.43). The transfer-difficulty ladder now reads: same-graph held-out
OD 1.59 (gen15) -> held-out CITY 1.68 (this) -> single-source cross-graph ~random (A2, the honest
boundary multi-city training removes). Caveats: 6 ODs per held-out graph, N=3 K=1, the FP-drift/
best-checkpoint discipline unchanged.
