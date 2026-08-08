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
