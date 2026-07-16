# Generation: gen12_sweeps (Obj-5 "varied levels of network disruption": the multi-convoy K / N / second-OD sweep tier)

- **status: PRE-REGISTERED 2026-07-10 (Kilian's overnight launch authority; night programme item 4);
  binding at launch.** Results appended below.
- **git SHA:** pinned by the commit landing this ledger (with the gen11 machinery + the vectorised
  `objective_matrix` closed forms that make the K=3 cells buildable, equivalence-tested).

## Why

Obj-5 promises evaluation "under varied levels of network disruption"; until tonight ONE instance,
ONE K and ONE N carried the whole multi-convoy claim, and no positive-arc result had a held-out
instance (CRITIQUE_PREFREEZE §1/§4). This tier produces the disruption CURVES (the F1/wave-A
design language: curves vs each cell's own oracle ladder, NO sacred-vs-vanilla gap gates) and the
held-out second OD in one stroke.

## Grid (12 runs; fleet-route; config = the gen11-SELECTED arm, recorded at launch below)

| axis | cells |
|---|---|
| headline OD **62-97 k8** | (N=3, K=1) x seeds {0,1,2}; (N=3, K=2), (N=3, K=3), (N=2, K=1), (N=5, K=1) x seed 0 |
| held-out OD **35-159 k8** | same five cells x seed 0 |

Second OD **35-159** selected tonight by the F3/SBO oracle dataset + verification probe, using the
gen09 screening criteria BEFORE any training: ratio loss_det/eq **3.39** (>= 3), leader asymmetry
H/lnR **0.44** (<= 0.85), R=12, and the STACKED-ONLY minimax optimum equals the full equilibrium
(0.206), so fleet-route mode is unhandicapped there (also verified for 62-97: 0.216 = 0.216).
Everything else identical to the gen11 arms: band 0.15-0.95, smooth FP tau 0.05, switch-every 200,
window 250, leader-ent-frac 0.5, floor 0.20, 1200 sorties, eval-every 100, EXACT estimator,
per-eval checkpoints, `--threads 3`, 3-parallel staged.

## Readout (PRE-REGISTERED: curves, NOT gap gates; the F1/wave-A discipline)

Per cell: the oracle ladder (shortest_path, ALNS = loss_det certificate, equilibrium) and SACRED's
exact best-checkpoint TAP (+ final TAP, disclosed drift). Reported as:
- **K curve at N=3** (both ODs): does SACRED's best-checkpoint track the equilibrium's growth in K
  (coverage saturating as K -> #routes) while remaining below ALNS?
- **N curve at K=1** (both ODs): does the margin over ALNS hold as the fleet grows (the oracle
  scan said the GAP grows with N; this is its first trained test)?
- **Held-out replication**: does the headline cell's qualitative ladder reproduce on an OD chosen
  by pre-registered screening criteria, not by outcome?
No STRONG-form bar is set for non-headline cells (they are single-seed curve points); the
qualitative claim per cell is SACRED(best-ckpt) < ALNS. Cells where K approaches route coverage
are expected to compress toward the (rising) equilibrium: that boundary is part of the curve.

## Commands (pinned; via `scratch/gen12_sweeps.sh`, detached, outputs under `models/runs/gen12_sweeps/`)

As the gen11 arms, with `--od {62-97, 35-159} --N {2,3,5} --K {1,2,3}` per the grid and the
gen11-selected arm flags (recorded at launch). Also in this batch (night item 3 completion):
post-fix VANILLA seeds 1, 2 on 62-97 (`--vanilla-only`), completing the 3-seed post-fix vanilla row.

## Launch record

- **gen11-selected arm flags: NONE (plain post-fix baseline).** No gen11 arm improved on the
  0.447 plateau (see `experiments/gen11_menuhead.md`), so the sweeps run on the unmodified
  fleet-route config, comparable to the standing post-fix numbers.
- git SHA: the commit landing this launch record; launched 2026-07-10 ~02:25 via
  `scratch/gen12_sweeps.sh` (ARM_FLAGS empty).

## RESULT (2026-07-10 03:34, 12 runs, ~71 min at 3-parallel staged): SACRED < ALNS in ALL 10 CELLS; the held-out OD reaches near-equilibrium POST-FIX

All values = exact best-checkpoint TAP (final TAP disclosed in the JSONs; drift persists everywhere,
as expected). Oracle ladders computed per cell (loss_det = the ALNS-certified deterministic optimum).

**Headline OD 62-97 k8:**

| cell | SACRED best-ckpt | ALNS (=loss_det) | equilibrium | SACRED/eq |
|---|---|---|---|---|
| N=3 K=1 (3 seeds) | **0.483 +/- 0.041** | 0.699 | 0.216 | 2.24x |
| N=3 K=2 | 0.759 | 0.907 | 0.432 | 1.76x |
| N=3 K=3 | 0.832 | 0.958 | 0.649 | 1.28x |
| N=2 K=1 | 0.341 | 0.690 | 0.190 | 1.79x |
| N=5 K=1 | 0.496 | 0.802 | 0.237 | 2.09x |

**Held-out OD 35-159 k8 (screened by pre-registered criteria BEFORE training):**

| cell | SACRED best-ckpt | ALNS (=loss_det) | equilibrium | SACRED/eq |
|---|---|---|---|---|
| N=3 K=1 | **0.261** | 0.699 | 0.206 | **1.27x** |
| N=3 K=2 | 0.500 | 0.866 | 0.412 | 1.21x |
| N=3 K=3 | 0.661 | 0.933 | 0.604 | **1.09x** |
| N=2 K=1 | 0.232 | 0.555 | 0.179 | 1.30x |
| N=5 K=1 | 0.389 | 0.719 | 0.230 | 1.69x |

Also in this batch: **post-fix vanilla completed to 3 seeds** (TAP 0.859 / 0.852 / 0.855 =
**0.855 +/- 0.003**; best-checkpoint 0.790-0.821): the control is tight and selection symmetry
does not rescue it.

**What the tier establishes:**
1. **Obj-5's "varied disruption" clause now has trained curves:** the qualitative claim
   (SACRED best-ckpt < ALNS = the deterministic-class optimum) holds in 10/10 cells across K in
   {1,2,3}, N in {2,3,5} and two instances; SACRED tracks the equilibrium's growth in K (coverage
   saturation) and the margin over ALNS GROWS with fleet size on the headline OD (N=5: 0.306 vs
   N=3: 0.216), the oracle scan's prediction confirmed in trained policies.
2. **The held-out replication is STRONGER than the headline instance:** every 35-159 cell lands at
   1.09-1.69x its equilibrium, POST-FIX, on honest representations, plain config, no gen11 terms.
3. **THE NIGHT'S KEY FINDING: the 0.447 plateau is INSTANCE-SPECIFIC to 62-97, not architectural.**
   On the more asymmetric held-out instance (leader H/lnR 0.44 vs 62-97's 0.63) the post-fix
   pipeline reaches 0.261 (1.27x eq) at the headline cell. Reading: 62-97's flatter, more
   route-symmetric equilibrium is exactly where the pre-fix identity-hash had been supplying the
   discrimination the mean-pooled head lacks; where the INSTANCE supplies asymmetry (a sharper FP
   gradient), honest embeddings suffice. This refines the gen10/gen11 attribution chain and is
   consistent with the campaign-long dogma that instance structure decides learnability.
4. Single-seed caveat on every non-headline cell (pre-registered as curve points, no gap gates).

**Consequence recommendation (for Kilian's morning decision, pre-registered nowhere = a NEW
decision):** 3-seed the ho_N3K1 cell (35-159, N=3, K=1). If it holds ~0.26 tight, it is the
natural candidate for THE post-fix multi-convoy headline (honest representations, instance
screened before training by the same criteria that picked 62-97, near-equilibrium), retiring the
two-headline pre-fix/post-fix asymmetry (CRITIQUE_PREFREEZE §2) entirely.

### DISJOINT-BASELINE APPENDIX (2026-07-16, Block R0; oracle/eval-only)

> **Context (binding wording rule; CRITIQUE_16-07-26.md §1; probes
> `scratch/disjoint_baseline_probe.py`, `scratch/r0_screen.py`, artefacts
> `models/runs/r0_screen.json`):** the candidate menus' first routes ARE the max-flow
> decomposition, and "uniform-stack over the edge-disjoint routes" (2 lines) is the strongest
> NAIVE baseline. No comparative sentence in this ledger may claim SACRED beats "every
> uncalibrated strategy class" or that its transfer is something "standard algorithms cannot
> achieve" without the rows below beside it.

Heuristic (uniform-disjoint-stack) rows for the 35-159 sweep cells, exact yardstick:
K=1 **0.250** (SACRED 0.261) | K=2 **0.494** (0.500) | **K=3 0.738 (SACRED 0.661: the FIRST
cell where trained calibration beats the strongest naive baseline; confirmed n=3 in gen26 step
1: 0.664 +/- 0.018)** | N=2 0.249 (0.232) | N=5 **0.250 (SACRED 0.389: the heuristic WINS the
fleet-shift cell)**. At K=4/5 the m=4 instance SATURATES for every defender (heuristic 0.966 /
0.985 ~ det 0.964 / 0.980, greedy yardstick) = the boundary's upper edge. **The honest curve:
learning pays in the band K = m-1 (measured) up to saturation; below it the heuristic suffices;
above it nobody wins.** gen26 carries the K >= m claim on an m=6 instance.
