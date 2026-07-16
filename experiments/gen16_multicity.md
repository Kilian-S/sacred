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

## ZERO-SHOT K/N ROWS (2026-07-11, item 2.4, EVAL-ONLY): the hedge survives budget/fleet shift

The frozen gen16 seed-0 best-checkpoint actor (trained at N=3, K=1) evaluated WITHOUT retraining on
held-out Gdansk ODs at shifted adversary budget and fleet size (best-checkpoint-centred TAP, the
gen16-fair estimator), scored against each (OD, K, N) cell's own oracle equilibrium:

| cell | gen (TAP) | random-init | beats loss_det |
|---|---|---|---|
| N=3 K=1 (train regime, sanity) | 1.71x | 1.99x | 6/6 |
| N=3 K=2 (zero-shot budget shift) | **1.29x** | 1.34x | 5/6 |
| N=5 K=1 (zero-shot fleet shift) | **1.79x** | 2.10x | 6/6 |

**The hedge survives both shifts:** the policy conditions on the MAP (edge vulnerability +
per-route features), not on K or N, so at an UNSEEN adversary budget (K=2) it tightens to 1.29x
(beats random, 5/6 beat loss_det) and at an UNSEEN fleet size (N=5) it holds at 1.79x (beats
random 2.10, 6/6 beat loss_det). The sanity row (N3K1 1.71x) matches the gen16 headline, confirming
the estimator. Zero-shot generalisation therefore extends beyond OD/city to the disruption axes
(K, N) the policy never trained on - the map-conditioning is the invariant.

## SCALE-AXIS EXTENSION: whole-Kyiv zero-shot row (2026-07-11, EVAL-ONLY, Kilian provided the map)

The frozen gen16 seed-0 best-checkpoint actor (selected by gen16's own held-out Gdansk, i.e.
select-on-train-derived, NOT tuned to Kyiv) evaluated zero-shot on the **whole Kyiv arterial
network (6083 nodes / 10861 edges** - the largest graph in the project, ~17x the training
Kaliningrad, built by the same extraction pipeline), 5 screened held-out ODs, single-checkpoint
exact evaluation:

| | mean ratio to eq | beats loss_det | per-OD gen / rand |
|---|---|---|---|
| generalist | **1.88x** | 3/5 | 1.33 / 1.77 / 1.63 / 2.68 / 1.97 |
| random-init | 2.03x | - | 2.02 / 2.18 / 2.09 / 2.41 / 1.47 |

**PARTIAL PASS:** generalist mean 1.88x <= 2.0 AND beats random-init (2.03x); beats loss_det on
3/5 ODs (the 2 misses are the thin-asymmetry ODs where loss_det/eq is only ~1.5-2x, so the
deployable margin is intrinsically small - the whole-project thin-headroom pattern, not a transfer
failure). **Honest deltas:** (1) single-checkpoint eval (the a2 harness), which understates vs the
best-checkpoint TAP the gen16 headline used - both arms are single-checkpoint so the beats-random
comparison is footing-fair; (2) Kyiv is less asymmetric than the training cities (eq med 0.253,
loss_det/eq med 1.96 vs the cities' 2.2-2.4), a genuinely harder transfer target. **What it adds:**
a whole large city, never trained on, transfers zero-shot beating random-init at 1.88x - the
scale axis of the transfer-difficulty ladder (held-out city Gdansk 1.68 best-ckpt -> whole-city
Kyiv 1.88 single-ckpt -> single-source cross-graph ~random). Route construction on the 6083-node
graph ran in seconds (the oracle is route/occupancy-bound, graph-size-independent at K=1).

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

**Selection disclosure (CRITIQUE_EXPANSION §4.2; dual-report from the saved JSONs):** best
checkpoint selected by the held-out-city mean (test-as-validation). Under select-on-TRAIN (the
honest alternative, held-out reported at the train-selected checkpoint): **select-on-test 1.677 +/-
0.072 vs select-on-train 1.733 +/- 0.149** (seed 1 moves 1.773 -> 1.941); final iterate 2.20. BOTH
pass the bar (1.733 <= 2.0, below the ~1.99 random-init reference); the loss_det clause is a
checkpoint-level property. The thesis dual-reports: lead with select-on-train 1.733 (deployable),
keep select-on-test 1.677 as the optimistic bound, and cite the final-iterate drift. Select-on-train
is the default for all subsequent generations.

**What is established (the aim-level ZST promise, at the CITY level):** one policy trained on
Kaliningrad + East London + Istanbul routes fleets in never-seen Gdansk at 1.68x its equilibria
zero-shot, beating the deterministic-class optimum on 17/18 cells, and generalises to a second
unseen graph (1.90 vs random 2.43). The transfer-difficulty ladder now reads: same-graph held-out
OD 1.59 (gen15) -> held-out CITY 1.68 (this) -> single-source cross-graph ~random (A2, the honest
boundary multi-city training removes). Caveats: 6 ODs per held-out graph, N=3 K=1, the FP-drift/
best-checkpoint discipline unchanged.

### DISJOINT-BASELINE APPENDIX (2026-07-16, Block R0; oracle/eval-only)

> **Context (binding wording rule; CRITIQUE_16-07-26.md §1; probes
> `scratch/disjoint_baseline_probe.py`, `scratch/r0_screen.py`, artefacts
> `models/runs/r0_screen.json`):** the candidate menus' first routes ARE the max-flow
> decomposition, and "uniform-stack over the edge-disjoint routes" (2 lines) is the strongest
> NAIVE baseline. No comparative sentence in this ledger may claim SACRED beats "every
> uncalibrated strategy class" or that its transfer is something "standard algorithms cannot
> achieve" without the rows below beside it.

Zero-shot rows on the SAME 6 held-out Gdansk ODs: **uniform-disjoint-stack 1.134x eq
(beats loss_det 6/6); inverse-vuln variant 1.024x eq** — with no training, no labels, no graph
exposure, no threat map. The gen16 generalist (1.677/1.733) does NOT beat this baseline; the
binding transfer wording is therefore: the generalist's zero-shot value is NOT superiority over
naive methods at K=1; it is (a) label-free learned amortisation bounded by the ladder
distill 1.555 < retrieval 1.676 < adversarial 1.733 < random ~1.99 < vanilla 2.354 with the
heuristic at 1.134 below all of them, and (b) the R0b structure row: zero-shot, the policy
concentrates 0.54-0.89 mass on each instance's disjoint core (eq allocates 0.53-0.97; uniform
~0.28) — it discovers the structure the heuristic must be told. The rescued transfer claims
live in gen27 (dynamic register) and gen26 (K >= m-1).
