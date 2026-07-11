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

**What it establishes (the culminating composite, on unseen ground):** the three pillars compose on
a city the policy never trained on - ZST makes each design's operational evaluation one forward
pass, SBO makes the search efficient, and the interdiction game supplies the objective; strategic
upstream design is optimised against the deployed operational system in a theatre no solver was
built for. The 0.109 vs 0.768 correlation contrast is the poster payload: the deployed policy's
operational landscape diverges from the equilibrium precisely where it matters most (an unseen
theatre), and this framework is the only one that can price it. This is also the concrete empirical
backbone of the ZST-vs-LP argument (CRITIQUE_EXPANSION §4.1): the LP's value proposition fails
exactly here, on the design loop over a deployed policy on a new map.
