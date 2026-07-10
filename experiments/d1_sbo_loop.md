# D1: the SBO acquisition loop over joint upstream design (placement x fleet size)

- **status: PRE-REGISTERED 2026-07-10 (expansion programme; ORACLE-ONLY, no policy training, so
  execution follows immediately under the probes-are-free rule); binding now.**
- **git SHA:** the commit landing this ledger + the loop script.

## Question (fixed before looking)

Does SURROGATE-GUIDED acquisition (the SBO loop proper: fit -> propose -> evaluate -> refit) find
near-optimal interdiction-aware designs with materially fewer exact evaluations than random
search at matched budget? (F3 proved the regression half; this is what makes it SBO rather than
supervised learning - the review's Forrester & Keane framing.)

## Design

- **Space:** ~900 designs (300 screened OD placements x N in {2,3,4}), k8 menus, band 0.15-0.95,
  K=1; objective = oracle `loss_mixed` (minimise = the most defensible design). The FULL space is
  oracle-evaluated once up front (cheap; seconds) SOLELY to know the true optimum for regret
  curves; the optimisers never see it.
- **SBO arm:** init n0 = 15 random designs; iterate to budget B = 60 total evaluations: fit a
  5-net SurrogateMLP ensemble (bootstrap seeds) on all evaluated designs (F3 features incl. the
  harmonic-vulnerability aggregate; train-stat normalisation); acquisition = LOWER CONFIDENCE
  BOUND (mu - kappa * sigma, kappa = 1.0) over unevaluated designs; evaluate the argmin; add.
- **Baselines:** random search (same budget) and the F3-style one-shot surrogate argmin (fit on
  n0, pick once). 20 repeats each (loop seeds 0-19).

## Decision metric (PRE-REGISTERED)

Simple regret r(b) = f(best design found by budget b) - f(true optimum), median over 20 repeats.
> **PASS:** median evaluations-to-regret <= 0.01 for the SBO arm is <= HALF random search's.
> **STRONG:** additionally the SBO arm's median final regret (b = 60) is 0.
Report the full regret-vs-budget curves for all arms.

## RESULT (2026-07-10, 900 designs, 20 repeats): PASS + STRONG

- Space: 900 designs (300 ODs x N in {2,3,4}); true optimum 0.099.
- **Median evaluations to regret <= 0.01: SBO 32.5 vs random INF** (random never reaches the bar
  within the 60-eval budget in the median). PASS bar (SBO <= half of random): MET (trivially, and
  by a wide margin).
- **Median final regret at budget 60: SBO 0.0000** (STRONG bar met: the SBO loop finds the exact
  optimum in the median repeat) **vs random 0.0449, one-shot F3-style 0.0583.** Per-repeat: SBO
  hit zero final regret in the large majority; both baselines plateaued ~0.045-0.058.

**What it establishes (Obj-4, the SBO loop proper, met):** surrogate-guided acquisition
(bootstrap-ensemble LCB) over the joint placement x fleet-size design space finds the optimal
interdiction-aware design in ~33 exact evaluations where random search does not reach the tolerance
in 60, and strictly dominates the one-shot surrogate argmin (the F3 regression) at every budget.
This is what makes it Surrogate-BASED OPTIMISATION rather than surrogate regression: the loop
(fit -> LCB-propose -> oracle-evaluate -> refit) is the review's Forrester & Keane framing realised.
Reproduce: `scratch/sbo_acquisition_loop.py`; artefact `models/runs/d1_sbo_loop.json`.
Future work (recorded): D2 (defender hardening budget as the tactical tier) and D3 (fit the
surrogate to the TRAINED generalist's exploitability rather than the oracle value; needs A1).
