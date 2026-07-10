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

## RESULT (to be appended)
