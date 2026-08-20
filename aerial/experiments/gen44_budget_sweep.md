# gen44_budget_sweep: authoring quality against search budget

Registered 2026-08-08, before any sweep call. Results 2026-08-09. No code SHA is pinned in the source ledger. Oracle and eval only, model proposals plus exact scoring, no training.
Artefacts: runner `analysis/gen44_budget_sweep.py`, per-configuration artefacts `models/runs/gen44_sweep/<config>.json` with full search histories and saved laydowns, marker `analysis/gen44_mark.py`.

## Question

Is there any search budget at which the authoring model's reasoning strength separates, and does that budget still produce a curriculum at or above the trainable knee of 0.022?

## Instrument

- Search: the step-5 / phase-1f authoring loop, imported rather than re-implemented (`llm_prompt`, `SCHEMA` and the exact `score` pool from `analysis/gen39_zeroshot.py`), model-parameterised so every configuration runs the identical loop.
- Operating point: budget 16 evaluations, up to 4 proposal rounds, temperature 0.9, doctrine frozen to DOC32, narva, K=3.
- Cells: three narva training fields (1000, 1001, 1002) x three repeats with distinct rng seeds, giving 9 searches per configuration. The running-best curve is recorded, so best-at-budget is read at b in {2, 4, 8, 16}.
- Configurations: llama-3.3-70b, Qwen3.6-27B thinking off and thinking on (gateway), and the four mounted Qwen3.5 rungs 2B, 4B, 9B and 27B (direct-to-port).
- Controls in the same sweep, no model involved: `local16` (hill-climb) and `random16` at identical budgets and repeats.
- Metric: irreducible threat, damage against a defender that knows the laydown, the same quantity as the step-5 mechanism table (llm16 0.0393, local16 0.0222, random16 0.0286, tuned 0.0278).

## Criteria

- Per configuration and budget, the median best-so-far over the 9 searches with a bootstrap 95% CI over searches.
- Separation test: at each budget, per-field paired bootstrap CIs on the difference between configurations. Separated only where the CI excludes zero.
- Knee row, always reported: the fraction of searches whose best-at-b falls below 0.022, per configuration and budget.
- Model arms reported per model, never pooled. Both directions reportable.

## Results

All 9 configurations completed 9/9 searches. Median best-at-budget over 9 searches:

| config | @2 | @4 | @8 | @16 |
|---|---|---|---|---|
| Qwen3.5-2B | 0.0119 | 0.0222 | 0.0286 | 0.0305 |
| Qwen3.5-4B | 0.0080 | 0.0245 | 0.0305 | 0.0341 |
| Qwen3.5-9B | 0.0278 | 0.0330 | 0.0343 | 0.0375 |
| Qwen3.5-27B | 0.0235 | 0.0375 | 0.0395 | 0.0431 |
| Qwen3.6-27B (thinking off) | 0.0287 | 0.0363 | 0.0381 | 0.0396 |
| Qwen3.6-27B (thinking on) | 0.0355 | 0.0358 | 0.0363 | 0.0385 |
| llama-3.3-70b | 0.0321 | 0.0321 | 0.0354 | 0.0370 |
| local16 (hill-climb) | 0.0214 | 0.0252 | 0.0282 | 0.0283 |
| random16 | 0.0167 | 0.0214 | 0.0264 | 0.0303 |

Paired separation tests:

| contrast | budgets where the CI excludes zero | at budget 16 |
|---|---|---|
| Qwen3.5-2B vs Qwen3.5-27B | 4, 8, 16 | 0.0305 against 0.0431 |
| Qwen3.5-4B vs Qwen3.5-27B | 4, 8, 16 | 0.0341 against 0.0431 |
| Qwen3.6-27B (thinking on) above hill-climb | 2, 4, 8, 16 | +0.0083 |
| llama-3.3-70b above hill-climb | 2, 4, 8, 16 | +0.0067 |
| Qwen3.5-27B vs Qwen3.6-27B (thinking off) | none | CI contains zero at every budget |
| thinking off vs on | 8 only, at -0.0034 favouring off | CI contains zero at 2, 4 and 16 |
| llama-3.3-70b vs Qwen3.6-27B (thinking on) | none | +0.0015, CI [-0.0015, +0.0049] |

Both size separations sit above the knee at the budgets where they hold.

Knee row: at budget 2, 56-78% of the weak configurations' searches fall below 0.022. From budget 4 upward every model configuration clears the knee on essentially every search.

Pre-registered branch: separation exists at usable budgets, so the branch recording a separating operating point fires. The separating pairings are size (2B or 4B against Qwen3.5-27B) and model against hill-climb, not generation and not thinking.
