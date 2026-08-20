# gen43_exam: forty-item placement exam for emplacement choice

Registered 2026-08-08, before bank construction and before any model call. Results 2026-08-08 (the seven papers), 2026-08-09 (the amendment rows). No code SHA is pinned in the source ledger. Eval only, no training.
Artefacts: bank `models/runs/gen43_exam/bank.json`, per-configuration artefacts and full traces in `models/runs/gen43_exam/` (including `qwen3-27b_think_t0.json`, `qwen3-27b_think_s1.json`, `qwen3-27b_think_s2.json`), builder `analysis/gen43_bank.py`, runner `analysis/gen43_exam.py`, marker `analysis/gen43_mark.py`.

## Question

Do model size, model generation, deliberation or model family change the quality of a K-slot emplacement choice, measured on one fixed paper with per-item pairing?

## Instrument

- Items: 40, each one a theatre, a seeded field, a seeded subset of S emplacement slots, and K teams. Theatres narva, kgd_gvardeysk, ukraine and fulda via the step-5 build (`base_for`); fields from the 43xxx seed range, disjoint from every gen39 train and test field.
- Difficulty: (S, K) cycles (6,2), (8,2), (8,3), (10,3), (12,3), giving 15 to 220 combinations per item.
- Ground truth: every combination of an item's subset valued exactly (`score_sites` irreducible threat, DOC32-frozen); ceiling is the maximum, the random reference is the median, and the full value table is stored in the bank.
- Validity screens, fixed before construction: at least S slots exist on the map and field, ceiling at least 0.010, ceiling at least 2x the median combination. A seeded candidate stream runs until 40 items pass and every screened-out candidate is recorded in the bank artefact.
- Prompt: the 1e catalogue text for the item's subset plus a fixed task asking for exactly K slot names, with the goal stated as maximising damage against a flight that already knows the positions. Answer schema `{"slots": [K names]}`. Prompts are built once at bank time and stored, so every configuration sees byte-identical papers.
- Format rule: one retry on a parse or format failure, after which the item is a format failure for that configuration, counting as unsolved and excluded from share of ceiling, with the exclusion count reported as a row.
- Configurations: Qwen3.5 rungs 2B, 4B, 9B and 27B (mounted, direct-to-port), Qwen3.6-27B thinking off and thinking on (gateway), llama-3.3-70b (gateway, cross-family reference, never in family statistics).
- Decoding: non-thinking arms at temperature 0, the thinking arm at temperature 0.6, seed 0, max_tokens 16000. One call per item per configuration.
- Marks: (a) mean share of ceiling over non-format-fail items, (b) items solved exactly, as a count, (c) mean percentile of the chosen combination within the item's full table.

## Criteria

- No superiority statement unless the per-item paired bootstrap CI excludes zero. Both directions of every contrast reportable.
- Format-fail counts are first-class rows, and the 2B is reported whatever its count.
- Contrasts fixed in advance: 4B vs 9B, 9B vs 27B, Qwen3.5-27B vs Qwen3.6-27B (generation), thinking off vs on, llama vs the 27Bs. Spearman of share against parameter count over the four Qwen3.5 rungs. Nothing pools with gen42 or gen39.
- Amendment criterion, fixed before the amendment calls: if the thinking arm's own seed spread is comparable to or larger than the thinking gap, the statement becomes that the gap sits within the arm's own sampling noise.

## Results

### Bank

40 items (narva 8, kgd 10, ukraine 11, fulda 11), 15 candidates screened out (5 with too few slots, 10 degenerate), 3,028 combinations valued exactly. Difficulty mix (S,K): (6,2) x8, (8,2) x10, (8,3) x8, (10,3) x9, (12,3) x5. Ceilings 0.0125 to 0.0662. Ceiling-to-median quartiles 2.9 / 5.5 / 14.2. Two of the 40 items have a near-zero median combination, legal under the screens and flagged in the artefact.

### Papers

Zero format failures anywhere in the act, 7/7 configurations, 40/40 items each.

| config | share of ceiling | solved /40 | mean percentile |
|---|---|---|---|
| Qwen3.5-2B | 0.483 | 3 | 0.664 |
| Qwen3.5-4B | 0.649 | 6 | 0.821 |
| Qwen3.5-9B | 0.719 | 6 | 0.868 |
| Qwen3.5-27B | 0.783 | 6 | 0.877 |
| Qwen3.6-27B (thinking off) | 0.830 | 11 | 0.912 |
| Qwen3.6-27B (thinking on) | 0.858 | 12 | 0.924 |
| llama-3.3-70b (off-ladder reference) | 0.627 | 4 | 0.806 |

Per-item paired contrasts on share, bootstrap 95% CI over 40 pairs:

| contrast | mean diff | CI | outcome |
|---|---|---|---|
| size 4B -> 9B | +0.070 | [-0.005, +0.157] | CI contains zero |
| size 9B -> 27B | +0.064 | [-0.023, +0.165] | CI contains zero |
| size cumulative 4B -> 27B | +0.134 | [+0.030, +0.246] | CI excludes zero |
| size cumulative 2B -> 27B | +0.300 | [+0.199, +0.405] | CI excludes zero |
| generation 3.5-27B -> 3.6-27B | +0.047 | [-0.017, +0.116] | CI contains zero |
| thinking off -> on | +0.028 | [-0.038, +0.093] | CI contains zero |
| llama-70B vs Qwen3.6-27B | +0.203 | [+0.097, +0.306] | CI excludes zero |
| llama-70B vs Qwen3.5-27B | +0.156 | [+0.057, +0.258] | CI excludes zero |

Spearman of share against parameter count over the four Qwen3.5 rungs: rho +1.000, exact permutation p 0.083, which is the floor at n=4 (24 orderings).

Solved exactly, tested pairwise by exact binomial on discordant items:

| contrast | discordant items | p |
|---|---|---|
| generation, 3.6-27B only vs 3.5-27B only | 6 vs 1 | 0.125 |
| thinking off vs on | 4 vs 3 | 1.000 |
| size 4B vs 27B | 4 vs 4 | 1.000 |
| llama vs Qwen3.6-27B | 1 vs 8 | 0.039 |

### Amendment: temperature control

Qwen3.6-27B, thinking on, at temperature 0, same bank and same prompts: 11/40 solved, mean share 0.8520, mean percentile 0.9251, zero format failures.

| paired contrast | mean diff | CI | outcome | answers differ |
|---|---|---|---|---|
| thinking off vs on, both at t=0 | +0.0222 | [-0.0487, +0.0892] | CI contains zero | 23/40 |
| thinking off (t=0) vs on (t=0.6) | +0.0280 | [-0.0399, +0.0923] | CI contains zero | 22/40 |
| thinking arm, t=0.6 vs t=0 | -0.0058 | [-0.0331, +0.0148] | CI contains zero | 5/40 |

### Amendment: seed repeats

The pinned thinking configuration repeated at seeds 1 and 2, zero format failures in all amendment rows:

| run | solved | mean share | mean percentile |
|---|---|---|---|
| seed 0 (pinned) | 12 | 0.8578 | 0.9244 |
| seed 1 | 10 | 0.8490 | 0.9188 |
| seed 2 | 12 | 0.8710 | 0.9265 |
| temperature 0 | 11 | 0.8520 | 0.9251 |

Seed range 0.0219, sd 0.0110 across three runs differing in nothing but the sampling seed.

Best estimate of the thinking effect, averaging the thinking arm over its three seeds and pairing per item against thinking off: +0.0295, CI [-0.0391, +0.0942], CI contains zero. Per seed: +0.0280, +0.0192, +0.0412, each straddling zero.

The temperature-0 row lands inside the seed range (0.8520 against 0.8490 to 0.8710). Pairwise answer-sheet differences across these comparisons are 5/40, 6/40 and 5/40.

The amendment criterion fires. The effect (0.0295) is of the same order as the arm's own seed range (0.0219), so on this instrument switching deliberation on moves the score by about as much as re-rolling the sampling seed does, while changing the answer on 23 of 40 items.
