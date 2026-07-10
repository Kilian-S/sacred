# Generation: gen14_evidence (C3: seed-strengthening + the missing 35-159 ladder rows)

- **status: PRE-REGISTERED 2026-07-10 (Kilian's expansion-programme launch authority: "add the
  rungs and launch C3"); binding at launch.**
- **git SHA:** the commit landing this ledger. All 20 headline-cell seeds run FRESH at this SHA
  (the existing seeds 0-2 of each cell were produced at earlier SHAs; the never-compare-across-
  git-states rule forbids pooling them, so the n=10 statistics stand entirely on this code state).

## Why (DIRECTION_EXPANSION C3)

n=3 with population std is the recorded statistical weak point of both headlines; the vanilla
control and the ALNS-forced-stack fairness row exist only on 62-97 (the retired instance), so the
LOCKED 35-159 ladder currently borrows them by analogy. This generation makes both headlines
n=10 with t-based 95% CIs and completes the 35-159 ladder natively.

## Arms

1. **MC cell:** gen13-lock config verbatim (35-159 k8 menu, band 0.15-0.95, N=3, K=1, fleet-route,
   smooth tau 0.05, switch 200, window 250, ent-frac 0.5, floor 0.20, 1200 sorties, eval-every
   100, exact estimator, per-eval ckpts), seeds {0..9}.
2. **SC cell:** gen10-SC config verbatim (33-71 k8 hard, walk, smooth, 3000 sorties, eval-every
   250, vanilla + sacred), seeds {0..9}.
3. **35-159 vanilla row:** `--vanilla-only` (independent convoys, travel objective), 1200
   sorties, seeds {0,1,2}.
4. **35-159 ALNS-forced-stack fairness row:** oracle-side (`classical_baselines`), eval-only.
5. **35-159 fleet-cost column:** exact best-checkpoint mixtures from the gen13 + gen14 ckpts
   (the `fleet_cost_probe` machinery), eval-only.

## Decision metrics (PRE-REGISTERED)

- **MC:** exact best-checkpoint TAP, mean +/- std AND t-based 95% CI over n=10; comparators ALNS
  0.699 / eq 0.206; expectation = consistent with the locked 0.274 +/- 0.025 (this is
  seed-strengthening, not a re-decision: the gen13 lock stands regardless; if n=10 lands
  materially worse it is reported and the lock is revisited by Kilian, not silently).
- **SC:** per-seed sacred vs vanilla TAP (sign count /10), pooled means, t-based CIs; comparators
  = the gen10-SC pooled 0.276 vs 0.480.
- Rows 3-5 reported as measured (no gates; they complete the ladder).

## Commands (pinned; `scratch/gen14_c3.sh`, detached, outputs `models/runs/gen14_evidence/`)

MC per seed: the gen13 command with `--seed $S`; SC per seed: the gen10-SC command with
`--threads 3`; vanilla row: `--vanilla-only`. 3-parallel waves; ~6 h wall total (SC dominates).

**Code-state disclosure (2026-07-10, before any results):** the A1 build landed the
edge-vulnerability observation column (EDGE_FEATURE_DIM 4 -> 5) while gen14's waves were queued,
so waves execute across `81d0dee` -> the bump commit. This is NON-BEHAVIOURAL for every gen14 arm:
all agents here are built at `edge_in_dim=4`, so `_clip_ea` slices the new column off and the
training inputs are byte-identical (the width-slicing back-compat mechanism, regression-tested;
suite 155 green at the bump commit). Disclosed rather than silent, per house rule.

## RESULT (to be appended)
