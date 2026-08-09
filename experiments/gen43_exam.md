# Generation: gen43_exam (the forty-question placement exam)

- **status: PRE-REGISTERED 2026-08-08, BEFORE any bank construction or model call (Kilian's
  in-conversation go). Eval-only, no training anywhere.**
- **git SHA at registration: this commit (aerial).**

## Question

gen42 ended with endpoints only: a 4B is worse than a 27B at the search-bound slot register,
and every finer contrast (middle steps, generation, thinking) drowned in n=8 sampling noise
(amendments 1-2). This act rebuilds the instrument for POWER rather than lengthening it:
many items, one answer each, every model on the same paper, per-item pairing. It asks the
gen42 questions again (size, generation, thinking, the cross-family reference) with resolving
power, and its fallback branch (still flat above 4B, now with tight intervals) is pre-declared
as citable as its climb branch.

## The instrument (pinned before construction)

- **40 items.** Each item = (theatre, seeded field, a seeded subset of S emplacement slots,
  K teams); theatres narva / kgd_gvardeysk / ukraine / fulda via the step-5 build (`base_for`);
  fields from the 43xxx seed range (disjoint from every gen39 train/test field); (S, K) cycles
  (6,2), (8,2), (8,3), (10,3), (12,3) for graded difficulty (15-220 combinations per item).
- **Ground truth:** EVERY combination of the item's subset valued exactly (`score_sites`
  irreducible threat, the 1e machinery's semantics, DOC32-frozen); ceiling = max, random
  reference = median; the FULL value table saved in the bank (percentiles need it).
- **Validity screens, pinned:** >= S slots exist on the (map, field); ceiling >= 0.010;
  ceiling >= 2x the median combination. The seeded candidate stream (item index drives map,
  S/K, field and subset rng) runs until 40 items pass; every screened-out candidate is
  recorded in the bank artefact.
- **Prompt** = the 1e catalogue text for the item's subset (correct v2 physics) + a fixed TASK
  asking for exactly K slot names, goal stated as maximising damage against a flight that
  already KNOWS the positions; answer schema `{"slots": [K names]}` enforced; the prompt is
  built ONCE at bank time and stored, so every config sees byte-identical papers.
- **Format rule:** ONE retry on parse/format failure, then the item is a FORMAT-FAIL for that
  config (counts as unsolved; excluded from share-of-ceiling with the exclusion count a
  first-class row).
- Builder `scratch/gen43_bank.py`; bank `models/runs/gen43_exam/bank.json`.

## Configurations and decoding (pinned)

qwen35-2b (expected format-limited, reported regardless), qwen35-4b, qwen35-9b, qwen35-27b
(mounted rungs, direct-to-port), crown qwen3-27b thinking OFF and ON (gateway),
llama-3.3-70b (gateway; cross-family reference, never in family statistics). Non-thinking
arms decode at temperature 0 (the deterministic modal answer; disclosed). The thinking arm
runs at its recommended operating point, temperature 0.6, seed 0, max_tokens 16000
(asymmetry disclosed). Runner `scratch/gen43_exam.py`, one call per item per config, full
traces banked per config.

## Marks (pinned)

(a) mean share of ceiling over non-format-fail items; (b) items solved EXACTLY (a count);
(c) mean percentile of the chosen combination among the item's full table (difficulty-free).
Contrasts, all per-item paired with bootstrap CIs: 4B vs 9B, 9B vs 27B, 3.5-27B vs crown-off
(generation), crown off vs on (thinking), llama vs the 27Bs (reference row). Spearman of
score vs parameter count over the 3.5 rungs. No superiority sentence below per-item paired
CI excluding zero; both directions of every contrast reportable.

## Binding rules

1. New instrument: nothing here pools with gen42 or gen39 numbers; they are motivation only.
2. Format-fail counts are first-class rows; the 2B is reported whatever its count.
3. Ops per the standing division (the stagehand mounts rungs; llama now runs under our own
   account, so the window needs no third party); every box call's trace banked locally.
4. Sequencing: bank build on the Mac now (the box is authoring step-5c); crown + llama papers
   tonight after authoring completes; the mounted rungs in one window after; marking on the
   Mac; the step-5c training is unaffected (different machine).

## RESULTS (appended per config; nothing above changes after results exist)

### BANK BUILT (2026-08-08, 21 min Mac, oracle-only; `models/runs/gen43_exam/bank.json`)

40 items (narva 8, kgd 10, ukraine 11, fulda 11), 15 candidates screened out (5 too few
slots, 10 degenerate), 3,028 combinations valued exactly. Difficulty mix (S,K): (6,2) x8,
(8,2) x10, (8,3) x8, (10,3) x9, (12,3) x5; ceilings 0.0125-0.0662; ceiling-to-median
quartiles 2.9 / 5.5 / 14.2. Disclosure: 2 of 40 items have a near-zero MEDIAN combination
(most answers score nothing, a few score; legal under the pinned screens and kept); the
percentile mark handles them cleanly, the share mark is unaffected, and they are flagged in
the artefact rather than replaced.

### ALL SEVEN PAPERS SAT (2026-08-08 20:06 - 22:31; marker `scratch/gen43_mark.py`;
### per-config artefacts + full traces in `models/runs/gen43_exam/`)

**Zero format failures anywhere in the act, 7/7 configurations, 40/40 items each.** The
paired redesign delivered the resolving power it was built for.

| config | share of ceiling | solved /40 | mean percentile |
|---|---|---|---|
| qwen35-2b | 0.483 | 3 | 0.664 |
| qwen35-4b | 0.649 | 6 | 0.821 |
| qwen35-9b | 0.719 | 6 | 0.868 |
| qwen35-27b | 0.783 | 6 | 0.877 |
| qwen3-27b (crown, off) | 0.830 | 11 | 0.912 |
| qwen3-27b (crown, thinking on) | 0.858 | 12 | 0.924 |
| llama-3.3-70b (off-ladder reference) | 0.627 | 4 | 0.806 |

**Per-item paired contrasts (share; bootstrap 95% CI over 40 pairs; SEPARATED = CI excludes
zero):**

| contrast | mean diff | CI | verdict |
|---|---|---|---|
| size 4B -> 9B | +0.070 | [-0.005, +0.157] | indistinguishable |
| size 9B -> 27B | +0.064 | [-0.023, +0.165] | indistinguishable |
| **size CUMULATIVE 4B -> 27B** | **+0.134** | **[+0.030, +0.246]** | **SEPARATED** |
| **size CUMULATIVE 2B -> 27B** | **+0.300** | **[+0.199, +0.405]** | **SEPARATED** |
| GENERATION 3.5-27B -> 3.6-27B | +0.047 | [-0.017, +0.116] | indistinguishable |
| THINKING off -> on (crown) | +0.028 | [-0.038, +0.093] | indistinguishable |
| **llama-70B vs crown** | **+0.203** | **[+0.097, +0.306]** | **SEPARATED** |
| **llama-70B vs 3.5-27B** | **+0.156** | **[+0.057, +0.258]** | **SEPARATED** |

Spearman(share vs parameter count) over the four Qwen3.5 rungs: **rho +1.000, exact
permutation p 0.083**, which IS the floor at n=4 (24 orderings), so the ordering is as strong
as four rungs can express and is reported as such, never as a conventional significance.
*(The marker's first pass printed scipy's asymptotic p of 0.000, which is meaningless at
rho=1, n=4; corrected to the exact permutation p before any reading was taken.)*

**Solved-exactly, tested pairwise (exact binomial on discordant items), because a count does
not wobble the way a small-sample median does:** generation 6 items solved only by the crown
vs 1 only by the 3.5-27B, p 0.125 (SUGGESTIVE, not licensed); thinking 4 vs 3, p 1.000;
size 4B->27B 4 vs 4, p 1.000; **llama vs crown 1 vs 8, p 0.039 (SEPARATED)**.

**Readings, and what they do to gen42.**
1. **Size genuinely helps, and the effect is cumulative rather than step-wise.** The ordering
   is perfectly monotone and both cumulative contrasts separate, while no single step does.
   gen42's surviving endpoint claim is CONFIRMED with resolving power and extended: the whole
   4B-to-27B span is real, not merely 4B-vs-27B at the extremes of a noisy instrument.
2. **Generation and thinking remain unresolved even at n=40 paired**, with both point
   estimates positive and small (+0.047, +0.028). The generation effect is suggestive on the
   harder exact-solve mark (p 0.125) and absent on thinking, so the licensed sentence stays
   "no measurable difference", now with tight intervals instead of gen42's helpless ones.
3. **NEW, and it overturns a gen42 clump:** llama-3.3-70b sits clearly BELOW both 27Bs on
   share (CIs exclude zero) and below the crown on exact solves (p 0.039). gen42 could not
   separate llama from the 27Bs at all; the powered instrument does. A 2024 70B is measurably
   worse at this task than 2026 27Bs, so neither parameter count nor vintage alone predicts
   it, but on this family-and-vintage pair the modern smaller models win.
4. **The 2B's format problem was an INSTRUMENT artefact, corrected here.** gen42 marked it
   FORMAT-LIMITED (11/16 and 5/8 valid replies); on this paper it returns 40/40 parsable
   answers and simply scores lowest. The gen42 wording ("cannot hold the answer format") is
   therefore scoped to that harness's schema and prompt, not to the model.
5. Binding rule 1 still holds: none of this pools with gen42 or gen39; those acts are
   motivation, and the share scale here is not comparable to their share-of-ceiling numbers.

**PROCESS INCIDENT, disclosed (found by the stagehand, repaired, and material).** A second,
ad-hoc track from another session ran the same three gateway papers to the same pinned output
paths. Its llama paper fired at ~21:50, precisely while llama was down for the mounting
window, took 40 x 502 Bad Gateway, and **overwrote the good llama artefact with a
format-fail-40, null-share row**. Nothing raised an alarm; it surfaced only when a summary
line threw on the null. Blast radius was llama only: the two crown artefacts were also
rewritten but byte-identically, since temperature 0 and a pinned seed make them reproducible.
Repaired 22:58-23:00 by re-running the pinned command unchanged; it reproduced the 20:08
result to every digit and `diff` against the surviving original log is empty, so all 40 items
agree on share, percentile and SOLVED. Both logs are preserved (`_ORIGINAL_2008`, `_rerun`).
Lesson recorded: pinned output paths plus two uncoordinated drivers is a silent-corruption
hazard, and a null summary field must be treated as an error rather than a datum.

### AMENDMENT (2026-08-08 23:2x, BEFORE the calls; overnight box work, eval-only)

The thinking contrast (+0.028, CI [-0.038, +0.093]) is the act's least settled row and carries
a disclosed asymmetry: the thinking arm decodes at temperature 0.6 with a pinned seed, every
other arm at temperature 0. Two additions, in priority order, both REPORTED rows that never
replace the pinned seed-0 paper and never enter the pre-registered contrasts above:

1. **Temperature control (the confound).** The crown, thinking ON, at temperature 0, same
   bank, same prompt. This separates deliberation from sampling temperature: if the thinking
   gap survives at matched temperature it is about reasoning, and if it vanishes it was the
   decoding asymmetry. Artefact `qwen3-27b_think_t0.json`.
2. **Noise floor for the one stochastic arm.** The pinned thinking configuration repeated at
   seeds 1 and 2, giving the arm's own run-to-run spread. The +0.028 gap is interpretable only
   against it. Artefacts `qwen3-27b_think_s1.json`, `_s2.json`.

Pre-committed reading: if the thinking arm's own seed spread is comparable to or larger than
+0.028, the licensed sentence hardens from "unresolved" to "within the arm's own sampling
noise", which is a stronger and more useful statement than a bare null. Both directions
reportable; nothing here reopens the pinned contrasts.

**Amendment plumbing (2026-08-08 23:5x).** The runner had temperature and seed as literals, so
the amendment rows were unrunnable. The stagehand was instructed NOT to edit the instrument and
correctly stood down rather than work around it (three workarounds considered and rejected in
its report, including a rewriting proxy and an edited copy). The flags were then added
deliberately by the analyst as `--temperature` / `--seed`, both defaulting to None, which
reproduces the pinned decoding exactly. **Inertness PROVEN, not asserted:** the crown
thinking-off paper was re-run through the patched runner and compared item by item against the
banked artefact, 40/40 items agreeing on status, choice, share, solved and percentile, zero
mismatching fields. The summary block now records the temperature and seed actually used, so
every future paper carries its own decoding provenance.

### AMENDMENT ROW 1 RESULT (2026-08-09 01:53, 67 min): THE TEMPERATURE CONFOUND IS RETIRED

The crown, thinking ON, at temperature 0 (`qwen3-27b_think_t0.json`; the runner's summary
records `temperature=0.0, seed=null`, so the flag genuinely applied): **11/40 solved, mean
share 0.8520, mean percentile 0.9251, zero format failures.**

| paired contrast | mean diff | CI | verdict | answers differ |
|---|---|---|---|---|
| thinking OFF vs ON, **both at t=0** (the control) | +0.0222 | [-0.0487, +0.0892] | indistinguishable | 23/40 |
| thinking OFF (t=0) vs ON (t=0.6), the banked pair | +0.0280 | [-0.0399, +0.0923] | indistinguishable | 22/40 |
| within the thinking arm, t=0.6 vs t=0 | -0.0058 | [-0.0331, +0.0148] | indistinguishable | 5/40 |

**Reading.** The disclosed decoding asymmetry was NOT carrying the thinking result: removing it
moves the gap from +0.028 to +0.022 and the two thinking decodings differ from each other by
-0.006 on only 5 of 40 items. **The thinking contrast is therefore unresolved for reasons of
effect size, not of instrument design**, which is the cleaner and more defensible position, and
the asymmetry caveat attached to every thinking sentence can be dropped.

**A finding in its own right.** Turning deliberation on changes the model's ANSWER on 23 of 40
items while changing its SCORE not at all. Thinking alters the path and not the destination,
which is the same shape as the gen39 repair's substitutes result (deliberation reaching, but
not exceeding, where feedback already arrives). Recorded as an observation with a measured
basis, not a mechanism.

The seed-repeat rows (the arm's own noise floor) follow at ~03:00 and ~04:07 and complete the
amendment.

### AMENDMENT ROWS 2-3 RESULT (2026-08-09 03:45, 04:11): THE NOISE FLOOR, AND THE PRE-COMMITTED
### BRANCH FIRES

The pinned thinking configuration repeated at seeds 1 and 2 (provenance recorded per row;
zero format failures in all three amendment rows):

| run | solved | mean share | mean percentile |
|---|---|---|---|
| seed 0 (banked) | 12 | 0.8578 | 0.9244 |
| seed 1 | 10 | 0.8490 | 0.9188 |
| seed 2 | 12 | 0.8710 | 0.9265 |
| temperature 0 (row 1) | 11 | 0.8520 | 0.9251 |

**The arm's own noise floor: seed range 0.0219, sd 0.0110** across three runs that differ in
nothing but the sampling seed.

**Best estimate of the effect,** averaging the thinking arm over its three seeds and pairing
per item against thinking-off: **+0.0295, CI [-0.0391, +0.0942], indistinguishable.** Per seed:
+0.0280, +0.0192, +0.0412, every one straddling zero.

**The pre-committed reading fires.** The amendment stated that if the arm's own seed spread is
comparable to the gap, the sentence hardens from "unresolved" to "within the arm's own sampling
noise". Effect 0.0295 against a seed range of 0.0219 is a ratio of 1.34, i.e. the same order.
**Licensed from now on: on this instrument, switching deliberation on moves the score by about
as much as re-rolling the sampling seed does.** That is a stronger and more useful statement
than a bare null, and it is what the act may say about thinking.

Row 1 lands INSIDE the seed range (0.8520 against 0.8490-0.8710), so temperature 0 versus 0.6
is likewise not separable from a seed re-roll, which independently confirms the row-1 verdict
that the decoding asymmetry was never carrying the result. Pairwise answer-sheet differences
are 5/40, 6/40 and 5/40 across all these comparisons, so roughly five to six items in forty is
simply what any re-roll moves.

**Standing correction to the marking artefact:** `marks.txt` was written by the night driver at
23:22, BEFORE any amendment row existed, so it covers the seven original papers only. The
amendment rows live in this section and in their own artefacts; the marker must be re-run before
`marks.txt` is cited for anything.
