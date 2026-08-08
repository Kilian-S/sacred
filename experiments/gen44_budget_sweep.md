# Generation: gen44_budget_sweep (does the author's reasoning strength ever separate, and at what search budget?)

- **status: PRE-REGISTERED 2026-08-08 ~20:35, BEFORE any sweep call. Oracle/eval-only: LLM
  proposals plus exact scoring, NO training anywhere, so it runs under the standing free-probe
  rule. Kilian's standing instruction tonight is that neither machine idles; this act is the
  work that fills the box after the gen43 papers.**
- **git SHA at registration: this commit (aerial).**

## Question

gen42 concluded, and gen39 step 5c assumes, that at a matched 16-evaluation budget every
author converges to the same curriculum strength, so a defender trained on one author's
curriculum should match another's. That conclusion currently rests on **one search run per
model** (the B-EFF curves, n=1, explicitly caveated in the gen42 ledger). This act puts error
bars on it and asks the sharper question the three-link argument raised: **is there ANY search
budget at which the author's reasoning strength separates, and does that budget still produce a
curriculum above the trainable knee (~0.022, from step 5's own mechanism table)?**

Answering it decides whether the deferred training comparison is worth four hours or is
predicted null with confidence. Both branches are useful and both are pre-declared below.

## Instrument (pinned)

- **Search:** the step-5 / phase-1f authoring loop, IMPORTED not re-implemented
  (`llm_prompt`, `SCHEMA`, and the exact `score` pool from `scratch/gen39_zeroshot.py`),
  model-parameterised so every configuration runs the identical loop. Budget 16 evaluations,
  up to 4 proposal rounds, temperature 0.9, doctrine frozen to DOC32, narva, K=3, the step-5
  operating point verbatim.
- **Cells:** three narva TRAINING fields (1000, 1001, 1002) x three repeats with distinct rng
  seeds = 9 searches per configuration. The running-best curve is recorded, so best-at-budget
  is read off at b in {2, 4, 8, 16} without extra cost.
- **Configurations:** llama-3.3-70b, crown qwen3-27b thinking OFF and ON (gateway), and the
  four mounted Qwen3.5 rungs 2B / 4B / 9B / 27B (direct-to-port, run inside their existing
  gen43 mounting window so no rung is mounted twice).
- **Metric:** irreducible threat (damage against a defender that knows the laydown), the same
  quantity step 5's mechanism table uses, so values are directly comparable with the banked
  curriculum-strength row (llm16 0.0393, local16 0.0222, random16 0.0286, tuned 0.0278).
- **Controls in the same sweep, no LLM involved:** `local16` (hill-climb) and `random16` at the
  identical budgets and repeats, since the whole question is whether an author beats dumb
  search, and the banked answer at 16 was that it does not.

## Marks and reading rules (pre-committed, both directions reportable)

1. Per configuration and budget: median best-so-far over the 9 searches, with a bootstrap 95%
   CI over searches. The n=1 readings that motivated this act are superseded by these, and the
   gen42 B-EFF rows are never pooled with them (different repeat structure).
2. **Separation test:** at each budget, per-field paired bootstrap CIs on the difference between
   configurations. A pair is SEPARATED only if its CI excludes zero.
3. **The decision the act exists for.** If no budget separates any LLM pair beyond the repeat
   noise, the licensed sentence becomes "curriculum authorship is insensitive to the author's
   reasoning strength at every tested budget, measured with repeats", and the deferred training
   comparison stays unrun with a firm rather than a predicted null. If some budget b* separates
   a pair AND both arms' curricula at b* sit at or above 0.022, that b* is recorded as the
   operating point any future training comparison must use, and the pairing is named.
4. **Knee row, always reported:** the fraction of searches whose best-at-b falls below 0.022,
   per configuration and budget, because a budget that discriminates by starving everyone below
   the trainable threshold answers nothing.
5. The LLM arms are reported per model, never pooled, per the standing rule.

## Artefacts

Runner `scratch/gen44_budget_sweep.py`; per-configuration artefacts
`models/runs/gen44_sweep/<config>.json` (full search histories, laydowns saved); marker
`scratch/gen44_mark.py`; ops interleaved with the gen43 mounting window by the stagehand.

## RESULTS (appended per configuration; nothing above changes after results exist)
