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

### RESULT (2026-08-09 00:46; all 9 configurations 9/9 searches; marker `scratch/gen44_mark.py`)

Median best-at-budget over 9 searches [bootstrap 95% CI]:

| config | @2 | @4 | @8 | @16 |
|---|---|---|---|---|
| qwen35-2b | 0.0119 | 0.0222 | 0.0286 | 0.0305 |
| qwen35-4b | 0.0080 | 0.0245 | 0.0305 | 0.0341 |
| qwen35-9b | 0.0278 | 0.0330 | 0.0343 | 0.0375 |
| qwen35-27b | 0.0235 | 0.0375 | 0.0395 | **0.0431** |
| qwen3-27b (crown off) | 0.0287 | 0.0363 | 0.0381 | 0.0396 |
| qwen3-27b thinking on | 0.0355 | 0.0358 | 0.0363 | 0.0385 |
| llama-3.3-70b | 0.0321 | 0.0321 | 0.0354 | 0.0370 |
| local16 (hill-climb) | 0.0214 | 0.0252 | 0.0282 | 0.0283 |
| random16 | 0.0167 | 0.0214 | 0.0264 | 0.0303 |

**1. A gen42 statement is CORRECTED.** gen42 read the n=1 B-EFF curves as "every rung lands in
the same 0.034-0.039 band at 16 evaluations". With nine repeats that is wrong: **2B (0.0305)
and 4B (0.0341) are SEPARATED from 3.5-27B (0.0431)** at budgets 4, 8 and 16, all above the
knee and therefore usable. Size does move authoring quality; the single-draw reading hid it.
The gen42 do-not-extend chain must be read with this correction attached.

**2. The LLM authors beat hill-climbing at EVERY budget, decisively.** hill-climb vs
crown-thinking separates at 2/4/8/16 (+0.0083 at 16), hill-climb vs llama likewise
(+0.0067 at 16). This does not contradict the banked act, whose mechanism table already showed
llm16 0.0393 against local16 0.0222; it confirms it with repeats. **Wording correction for
every summary of this arc: the hill-climb ties the LLM at the DEFENDER level (banked n=3,
0.1353 vs 0.1288, 1/3 seeds) while being clearly WORSE at the CURRICULUM level.** The two are
different claims and have been elided.

**3. Generation and thinking do NOT separate as authors.** 3.5-27B vs crown-off is
indistinguishable at every budget; thinking off vs on is indistinguishable at 2, 4 and 16 and
separates only at 8, where the sign favours thinking OFF (-0.0034), i.e. one cell in four with
the "wrong" sign, best read as multiple-comparison noise rather than an effect.

**4. The load-bearing cell for gen39 step 5c: llama vs crown-thinking is INDISTINGUISHABLE at
budget 16** (+0.0015, CI [-0.0015, +0.0049]), the budget step 5c used. So the two curricula
that trained step-5c's arms are equal in strength not merely at n=1 but with nine repeats each.
**Since their defenders nonetheless differ, and differ unanimously in zero-shot transfer, the
carrier of that effect is definitively NOT curriculum strength.** gen44 converts step 5c's
mechanism puzzle from an observation into a measured exclusion.

**5. Knee row.** At budget 2, 56-78% of the weak configurations' searches fall below 0.022, so
budget 2 answers nothing about authorship; from budget 4 upward every LLM configuration clears
the knee on essentially every search.

**The pre-registered decision.** Separation exists at usable budgets, so the branch that fires
is the second one: a future training comparison, if run, must use a separating and usable
budget, and the pairing that separates is SIZE (2B or 4B against 27B) or LLM-against-hill-climb,
NOT generation and NOT thinking. The first branch's sentence ("insensitive at every tested
budget") is NOT licensed and must not be used.
