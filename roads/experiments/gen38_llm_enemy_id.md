# gen38: LLM enemy-doctrine identification
Registered 2026-07-24. Results 2026-07-24 (V1 and robustness row), 2026-07-25 (V2). Code: V2
batch at SHA `475698b`.
Artefacts: `models/runs/gen38_llm_enemy_id/v1_result.json`,
`models/runs/gen38_llm_enemy_id/transcripts/`,
`models/runs/gen38_llm_enemy_id/robustness.json`,
`models/runs/gen38_llm_enemy_id/v2_seed{0,1,2}.{json,log}` plus checkpoints,
`models/runs/gen34_hidden_adversary/family_refs.json`. Scripts
`analysis/gen38_enemy_id.py`, `analysis/gen38_narratives.py`, `analysis/gen38_robustness.py`,
`analysis/gen38_v2_batch.sh`, `analysis/gen34_family_probe.py`, `analysis/dyn_exact.py`,
`scripts/train_family_generalist.py`.

## Question
Can a language model reading a behavioural intelligence assessment identify the hidden enemy
doctrine well enough that deploying the matching counter crosses the exact type-blind wall
measured in gen34?

## Game
- Apparatus inherited from gen34: the five-member doctrine family, six held-out Gdansk cells,
  each member's exact optimal specialist policy (`greedy_policy_from_rvi`), per-type omni
  values, the omni cap (perfect type identification) and the blind cap (best type-blind play).
- Narratives: 20 hand-authored intelligence assessments, 4 per member, behavioural wording, the
  type never named, committed verbatim in `analysis/gen38_narratives.py` and never edited after
  any LLM output was seen.
- Classifier: `llama-3.3-70b` shown the five doctrine descriptions plus one narrative, returning
  {type, confidence 0-1, reasoning} as JSON; temperature 0.2; 3 independent draws per narrative
  (60 draws). Full transcripts committed.
- V1 operational eval, oracle-exact: the assisted defender plays specialist[predicted] against
  the true type; per held-out instance the value is the mean over the 20 narratives of
  `policy_value_exact(specialist[pred], cost[true])`, pooled over the 6 cells.
- V2: one type-conditioned SACRED policy trained under the gen34 pool and config, 3 seeds x
  12000 sorties, type delivered as a per-route type-threat column, flag `--type-conditioned`
  in `scripts/train_family_generalist.py`; then deployed with the LLM-supplied type at eval.
- Robustness row: narratives degraded programmatically, not re-authored. TERSE = first sentence
  only; DISTRACTOR = full assessment plus one conflicting sentence from a different doctrine;
  BOTH = terse plus distractor.

## Criteria
- PRIMARY: pooled commit-to-argmax operational value below the blind cap on >= 4/6 held-out
  cells.
- STRONG: pooled operational value <= omni_cap x 1.15.
- COMPARATIVE: LLM classification accuracy above the keyword baseline's, and LLM operational
  value below the keyword operational value.
- Reported, not gated: the 5x5 confusion matrix, confidence calibration, the commit-versus-hedge
  delta, and the majority-vote-over-3 row.
- V2 is gated on a V1 pass.

## Baselines
- Keyword control: fixed keyword-to-type table, committed.
- Random control: uniform 5-way, analytic expectation.
- Blind cap: best type-blind play, the wall; gen34's trained type-blind generalist reached
  1.373x it.
- Omni cap: value of playing each type's exact specialist with the type known.
- Confidence-gated variant: play the blind policy when confidence < 0.5, reported beside the
  commit-to-argmax primary.

## Results

### V1 (2026-07-24, oracle plus LLM, no training)
Held-cell caps reproduced before any new number was read (249-95 blind_cap 0.1198 / omni_cap
0.0589).

| quantity | value |
|---|---|
| LLM classification accuracy | 1.000 (60/60 draws; perfect diagonal confusion) |
| keyword-control accuracy | 0.800 |
| random-control accuracy | 0.200 |
| blind_cap (the wall) | 0.1140 pooled |
| omni_cap (perfect type identification) | 0.0631 pooled |
| LLM commit-to-argmax | 0.0631 pooled = omni_cap; crosses the wall 6/6 cells |
| keyword commit | 0.1942 pooled; crosses 0/6 |
| random commit | 0.2925 pooled |

Criterion outcomes: PRIMARY met, 6/6 against a bar of >= 4/6. STRONG met (<= 1.15x omni_cap).
COMPARATIVE met (accuracy 1.000 > 0.800 and value 0.0631 < 0.1942). The keyword control's mean
0.194 sits above blindness 0.114.

### Robustness row (2026-07-24)
Blind_cap 0.1140, omni_cap 0.0631.

| condition | LLM acc | keyword acc | LLM op value | keyword op value | LLM crosses wall |
|---|---|---|---|---|---|
| clean (V1) | 1.000 | 0.800 | 0.0631 | 0.1942 | 6/6 |
| terse | 0.950 | 0.800 | 0.1035 | 0.1758 | 5/6 |
| distractor | 0.800 | 0.450 | 0.0895 | 0.3701 | 6/6 |
| both (terse+distractor) | 0.400 | 0.400 | 0.2175 | 0.3659 | 0/6 |

### V2 (2026-07-25, type-conditioned SACRED, 3 seeds x 12000 sorties)
| seed | best told-TRUE-type (ratio to blind cap) | told-LLM-type at that checkpoint |
|---|---|---|
| 0 | 0.670 @ 8000 | 0.673 |
| 1 | 0.664 @ 7000 | 0.663 |
| 2 | 0.657 @ 12000 | 0.660 |
| pooled | 0.664 | 0.665 |

Criterion outcome: the type-conditioned policy is below the type-blind wall on 3/3 seeds,
pooled 0.664x the blind cap against the gen34 type-blind generalist's 1.373x. Pooled achieved
value 0.0758, a capture of about 75% of the exact inference gap ((0.1140 - 0.0758) /
(0.1140 - 0.0631)). The LLM-supplied type and the true type differ by 0.001-0.003 pooled across
seeds.
