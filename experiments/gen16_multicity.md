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

## RESULT (to be appended)
