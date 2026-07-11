# Generation: gen23_c1 (C1: ERB bootstrapping via a population-based metaheuristic, Obj-3 verbatim)

- **status: PRE-REGISTERED 2026-07-11 (expansion item 3); chained after gen22; binding now.**

## Why

Obj-3 verbatim: "investigate the efficacy of ERB bootstrapping via population-based metaheuristics
to accelerate training convergence". gen01 left this inconclusive at n=1 on the campaign problem.
This closes the wording on the post-fix interdiction headline: does seeding the replay buffer with
ALNS-population demonstrations accelerate the fleet-route defender's convergence?

## Design

Seed the fleet-route defender's buffer with demonstration transitions of stacking on the routes a
POPULATION of ALNS plans (8 restart seeds) favours (ALNS minimises worst-case mission-failure, so
its routes are the equilibrium's low-vulnerability support), scored vs the equilibrium attacker.
Arms {seeded (`--erb`), cold} x seeds {0,1,2} on 35-159, otherwise the gen14 fleet-route config
(smooth FP tau 0.05, 1200 sorties, eval-every 100). `--erb-copies 200`.

## Decision metric (PRE-REGISTERED)

Anchors: headline best-ckpt TAP 0.256; competence bar TAP <= 0.35 (comfortably above the headline,
a "reaches a competent hedge" threshold).

> **PRIMARY (time-to-competence): seeded reaches TAP <= 0.35 in FEWER sorties than cold**, mean
> over 3 seeds (the acceleration claim). **SECONDARY: final best-checkpoint parity** (seeded and
> cold both reach ~0.256, i.e. ERB accelerates without changing the ceiling). Either outcome closes
> the verbatim wording: acceleration (positive) or "ERB does not accelerate on this game" (honest
> null, n=3, post-fix, definitive vs gen01's n=1).

## RESULT (2026-07-11, {seeded, cold} x 3 seeds): NEGATIVE - ERB from a metaheuristic HURTS, and the mechanism is thesis-relevant

| arm | time-to-competence (TAP <= 0.35) | best-ckpt TAP |
|---|---|---|
| COLD | 500 / 100 / 100 sorties | 0.321 / 0.246 / 0.287 (mean 0.285) |
| SEEDED (ALNS demos) | never / never / never | 0.466 / 0.419 / 0.442 (mean 0.443) |

> **The seeding HURTS: cold reaches competence in ~100-500 sorties and lands at 0.285 (near the
> headline 0.256); seeded NEVER reaches the 0.35 bar and plateaus at 0.443.** PRIMARY (seeded
> faster): FAIL. This is a definitive n=3 post-fix result (vs gen01's n=1 inconclusive).

**What is established (Obj-3 verbatim wording closed, with a mechanism):** ERB bootstrapping via a
population-based metaheuristic does NOT accelerate convergence on the interdiction headline - it
DEGRADES it - and the reason is exactly the thesis's central theme: **ALNS produces DETERMINISTIC
plans (it minimises worst-case mission-failure to loss_det = a fixed spread), so its demonstrations
teach committed, single-route behaviour; seeding a MIXED-STRATEGY learner (whose optimum is
randomise-your-stack) with deterministic expert demonstrations biases it toward exploitable
determinism, away from the equilibrium randomisation.** The very determinism that makes the
metaheuristic a good deterministic planner makes its demonstrations counterproductive for a game
whose solution is a mixed strategy. So the honest Obj-3 conclusion: ERB-from-metaheuristic is the
wrong tool for a security game; demonstration bootstrapping helps only when the demonstrated
behaviour matches the target solution concept (the gen09 forced-copy-from-a-MIXING-leader arc,
which DID help, is the contrast). This is a genuine, mechanistic finding, not just a null.
