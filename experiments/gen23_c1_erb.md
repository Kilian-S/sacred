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

## RESULT (to be appended)
