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

## RESULT (2026-07-10)

### MC headline 35-159, n=10 (best-ckpt TAP): TIGHTER and consistent with the lock

Per-seed: 0.238, 0.244, 0.248, 0.248, 0.251, 0.255, 0.260, 0.264, 0.267, 0.285.
> **mean 0.256, sd 0.014, 95% t-CI [0.246, 0.266]** (all fresh at this SHA). The gen13 n=3 lock
> (0.274 +/- 0.025) sits inside; n=10 is if anything slightly better and 1.8x tighter. Headline
> ladder holds with a real CI: shortest 0.912 > ALNS 0.699 > **SACRED 0.256 [0.246, 0.266]** >
> equilibrium 0.206 (1.24x eq, 2.7x below ALNS). The n=3 lock stands; this is the citable CI.

### Native 35-159 ladder rows (completing the headline instance; no more borrowing from 62-97)

| arm | mission-failure exploitability | note |
|---|---|---|
| shortest_path | 0.912 | naive stack |
| ALNS forced-to-STACK (fairness) | 0.841 | ALNS is free to stack but SPREADS by choice |
| ALNS (spread, = loss_det) | 0.699 | the deterministic-class optimum |
| uniform-INDEPENDENT | 0.546 | oracle row (2026-07-12, `scratch/uniform_stack_probe.py`) |
| vanilla (non-adversarial SAC, n=3) | best-ckpt **0.526** / final 0.628 +/- 0.006 | see note |
| uniform-STACK (one uniformly-random route) | 0.442 | the strongest NAIVE-randomisation heuristic (oracle row, 2026-07-12) |
| **SACRED (adversarial, n=10)** | **0.256 [0.246, 0.266]** | the headline |
| equilibrium (loss_mixed) | 0.206 | computable bound |

**Naive-randomisation rows (added 2026-07-12; CRITIQUE_EXAMINER.md §5.1):** the ladder now bounds
SACRED against every uncalibrated strategy class: deterministic (ALNS 0.699), independent mixing
(0.546), incidental learned mixing (vanilla 0.526), and stack-and-randomise-uniform (0.442).
SACRED's margin over the best of them is 0.186 (42% relative): the claim is calibrated
randomisation, not randomness. The 62-97 (retired pre-fix instance) analogues for the historical
record: uniform-independent 0.848, uniform-stack 0.649.

**Honest ordering note (instance-dependent, chapter-worthy):** on 35-159 the non-adversarial
vanilla (0.526 best-ckpt) sits BELOW ALNS (0.699) - the reverse of 62-97 (vanilla 0.855 > ALNS).
This is not an error: ALNS emits a DETERMINISTIC plan (one occupancy, fully exploitable to its
worst-case interdiction -> 0.699), whereas even non-adversarial SAC emits a STOCHASTIC occupancy
distribution, so its incidental mixing is a (poorly-calibrated) mixed strategy that on this
asymmetric instance already beats the deterministic optimum. It is the milder version of the whole
thesis: stochasticity buys unexploitability; ADVERSARIAL training buys the CALIBRATED mixing that
gets to 0.256. **SACRED beats vanilla by 0.27 (best-ckpt) and beats a control that itself beats
ALNS** - if anything a stronger Obj-5 result than the 62-97 ordering. The forced-stack fairness row
(0.841 >> ALNS spread 0.699) reproduces natively: ALNS spreads by choice, so SACRED's win is the
randomisation, not a denied stacking privilege.

### SC headline 33-71, n=10: the pre-registered primary now has a paired CI excluding zero

Per-seed sacred TAP: 0.243, 0.260, 0.266, 0.296, 0.297, 0.304, 0.327, 0.329, 0.379, 0.394.
> **sacred mean 0.310, 95% t-CI [0.275, 0.345]; vanilla mean 0.485; sacred < vanilla on 10/10
> seeds; paired dD (vanilla - sacred) mean 0.175, 95% CI [0.137, 0.213] EXCLUDING ZERO.** The
> gen10-SC n=3 reading (sacred 0.276, vanilla 0.480) sits inside; the n=10 sacred mean is a touch
> higher (0.310) with wider spread (the FP-cycling tail: two seeds at 0.38-0.39), but the PAIRED
> comparison - the pre-registered primary - is now significant at n=10, not merely 3/3 signs.
> Ladder holds: shortest 1.000 > vanilla 0.485 > uniform 0.455 > **sacred 0.310** >> equilibrium
> 0.167.

## gen14 CLOSED (2026-07-10 17:20). Both headlines now carry n=10 CIs:
- **multi-convoy 35-159: best-ckpt TAP 0.256, 95% CI [0.246, 0.266]** (2.7x below ALNS, 1.24x eq);
- **single-convoy 33-71: sacred 0.310, paired dD vs vanilla 0.175 [0.137, 0.213] excl. 0, 10/10;**
- 35-159 native vanilla + forced-stack rows recorded; the n=3 locks stand, these are the citable CIs.
The statistical weak point flagged in CRITIQUE_PREFREEZE §4.3 is closed.

### DISJOINT-BASELINE APPENDIX (2026-07-16, Block R0; oracle/eval-only)

> **Context (binding wording rule; CRITIQUE_16-07-26.md §1; probes
> `scratch/disjoint_baseline_probe.py`, `scratch/r0_screen.py`, artefacts
> `models/runs/r0_screen.json`):** the candidate menus' first routes ARE the max-flow
> decomposition, and "uniform-stack over the edge-disjoint routes" (2 lines) is the strongest
> NAIVE baseline. No comparative sentence in this ledger may claim SACRED beats "every
> uncalibrated strategy class" or that its transfer is something "standard algorithms cannot
> achieve" without the rows below beside it.

Rows for BOTH n=10 ladders:
- **35-159 (MC):** uniform-disjoint-stack **0.250** (cost 99.5), inv-vuln **0.241** — inside /
  below SACRED's CI [0.246, 0.266]. SACRED's surviving edges: fleet cost 90.4 vs 99.5 (eq 91.0)
  and near-equilibrium structural allocation (R0b: mass 0.62 vs eq 0.70 vs uniform 0.33).
- **33-71 (SC, hard K=1):** uniform over the 6 disjoint routes = **0.167 = the exact
  equilibrium** (m=6; the menu-uniform 0.455 anchor is a padded-menu row). Every trained SC
  number (0.310 [0.275, 0.345] here; 0.276 gen10-SC; 0.362 B2-P3) sits above it. The SC act's
  citable content is the sacred-vs-vanilla PAIRED contrast (dD 0.175 [0.137, 0.213]) and the
  learning-dynamics account, NOT proximity to the equilibrium.
