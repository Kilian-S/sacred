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

## RESULT (to be appended)
