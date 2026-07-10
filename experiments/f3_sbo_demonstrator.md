# F3: the Obj-4 SBO demonstrator (interdiction-aware placement + fleet sizing, reduced form)

- **status: DONE 2026-07-10 (night programme item 2; EVAL-ONLY, no policy training).** Script:
  `scratch/sbo_placement_demo.py`; artefact `models/runs/sbo_placement_demo.json`; SHA = the
  commit landing this ledger.

## Question (fixed before looking)

Can a NEURAL METAMODEL (the repo's `SurrogateMLP`) approximate the interdiction-aware quality of a
base/FOB PLACEMENT + FLEET SIZE design well enough to select good designs without solving the
game, validated on held-out placements? (Lit-review Obj-4's "neural network metamodel to
approximate facility location and fleet composition", in reduced form: the exact evaluator being
surrogated is the equilibrium ORACLE, not a trained policy; the full SBO loop is future work.)

## Design

- **Design space:** 150 candidate OD placements (high-connectivity pairs, 3-6 disjoint routes,
  k8 shared-edge menus, soft band 0.15-0.95, K=1, absolute norm) x fleet size N in {2,3,4} =
  **450 designs**; exact objective per design = oracle `loss_mixed` (equilibrium mission-failure =
  how defensible the placement is under optimal randomised play; the whole dataset costs 8 s).
- **Features (cheap, pre-solve, no LP at query time):** OD shortest-path distance, route count,
  candidate-edge count, route-cost stats, edge-vulnerability stats, mean pairwise route-overlap
  Jaccard, N, plus a THEORY-GUIDED aggregate: the harmonic aggregate of per-route worst
  vulnerabilities 1/sum_r(1/p*_r) (= the closed-form disjoint-route single-convoy equilibrium;
  shared edges and the mission objective bend it, which is what the MLP learns).
- **Validation split BY OD PAIR** (a placement's designs never straddle train/test): 339 train /
  111 test designs (37 held-out placements). Metric fixed before looking: held-out RMSE, Spearman
  rank correlation, and ARGMIN VALIDATION (the surrogate's chosen design's true value vs the true
  best = regret).

## RESULT (2026-07-10)

> **Held-out: RMSE 0.0222 (target range 0.099-0.332) | Spearman rho 0.894 (p = 7e-40) |
> argmin regret 0.0000** - the surrogate's chosen placement (11-127, N=2, true 0.132) attains the
> true-best equilibrium exploitability (9-127, N=2, 0.132). Top-5 overlap 2/5 (the design space
> has many near-ties at the top; the regret metric is the operative one).

**What it establishes (Obj-4, reduced form, met):** a small neural metamodel predicts a
placement/fleet design's equilibrium mission-failure from cheap structural features accurately
enough to rank designs (rho 0.89) and to SELECT an optimal design on held-out placements with zero
regret, replacing the exact game solve at query time. Honest scope: the surrogate approximates the
ORACLE's evaluation (facility location + fleet composition against the interdiction equilibrium);
coupling it to the TRAINED policy's exploitability, and the full SBO acquisition/refinement loop,
are future work (one sentence in the thesis).

## Reproduction

```bash
PYTHONPATH=. .venv/bin/python scratch/sbo_placement_demo.py
```
First pass (60 pairs, no harmonic feature) recorded for honesty: RMSE 0.0422, rho 0.522, regret
0.0573; the improvement came from 3x more oracle data (the oracle is cheap: that is the point of
this design space) + the closed-form-guided feature, both decided on train-side evidence only.
