# D3-on-Gdansk: the composite exhibit on a NEVER-TRAINED city (expansion item 5, promoted)

- **status: DONE 2026-07-11 (EVAL-ONLY; `scratch/d3_gdansk.py`; artefact `models/runs/d3_gdansk.json`).**

## What

The D1 SBO acquisition loop over (base placement x fleet size) designs ON THE HELD-OUT CITY
(Gdansk), where each design's objective is the frozen MULTI-CITY generalist's operational
exploitability (one forward pass + one oracle BR per design). Strategic design in a theatre the
policy NEVER trained on, priced entirely by zero-shot transfer, in a loop no LP can enter.

## RESULT

- **Surrogate over the TRAINED generalist's exploitability, on the never-trained city: held-out
  Spearman 0.862.** The metamodel predicts the deployed zero-shot policy's vulnerability on unseen
  Gdansk designs well enough to optimise over them.
- **SBO acquisition: median 32.5 evaluations to regret <= 0.01 vs random INF; median final regret
  0.0000 vs random 0.0107.** The loop finds the operationally-optimal design efficiently.
- **policy-vs-oracle design-target rank correlation: 0.109** (vs 0.768 for the in-distribution D3).
  On an UNSEEN theatre, designing against the deployed policy is almost UNCORRELATED with designing
  against the equilibrium abstraction - the gap the in-distribution D3 found (0.768) WIDENS on a
  transfer city. So on a new theatre you MUST design against the deployed policy, and only the
  RL + surrogate loop can: the LP would re-solve the equilibrium and hand you a design that is
  near-irrelevant to how the zero-shot policy actually performs there.

## A5 RELIABILITY CHECK (2026-07-12, EVAL-ONLY; `scratch/d3_gdansk_reliability.py`): the 0.109 is NOT seed-stable; the poster claim is DOWNGRADED per the pre-committed gate

The design-target evaluator is exact, so the reliability axis is CROSS-SEED: the same 165-design
sweep re-priced under all three independently-trained gen16 actors (each at its selected best
checkpoint). Artefact: `models/runs/d3_gdansk_reliability.json`.

| quantity | value |
|---|---|
| cross-seed design-ranking Spearman (0-1 / 0-2 / 1-2) | 0.491 / 0.321 / 0.543 (mean **0.451**, gate >= 0.5: MISSED) |
| policy-vs-oracle target Spearman per seed | **0.109 (seed 0) / 0.443 (seed 1) / 0.433 (seed 2)** |

**Consequences (binding for poster + storyline):** (1) the headline 0.109 was substantially a
SEED-0 artefact; the honest cross-seed statement is "policy-vs-oracle design-target correlation
0.11-0.44 on the unseen city, vs 0.768 in-distribution": still a real gap, no longer a dramatic
near-zero. (2) Cross-seed reliability 0.32-0.54 means the operational design landscape is
POLICY-INSTANCE-specific: two equally-good adversarially-trained policies induce materially
different design rankings on an unseen theatre. Stated carefully this remains an argument FOR the
composite loop (you must price THE policy you will deploy, not a class-average and not the
equilibrium abstraction), but the exhibit's specific numbers must be presented per-seed with the
reliability disclosed, and the "almost uncorrelated" wording is retired. (3) The in-distribution
D3 result (0.768, Spearman 0.959 surrogate) is unaffected.

**What it establishes (the culminating composite, on unseen ground):** the three pillars compose on
a city the policy never trained on - ZST makes each design's operational evaluation one forward
pass, SBO makes the search efficient, and the interdiction game supplies the objective; strategic
upstream design is optimised against the deployed operational system in a theatre no solver was
built for. The 0.109 vs 0.768 correlation contrast is the poster payload: the deployed policy's
operational landscape diverges from the equilibrium precisely where it matters most (an unseen
theatre), and this framework is the only one that can price it. This is also the concrete empirical
backbone of the ZST-vs-LP argument (CRITIQUE_EXPANSION §4.1): the LP's value proposition fails
exactly here, on the design loop over a deployed policy on a new map.
