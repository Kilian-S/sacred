# Generation: gen38_llm_enemy_id (Phase-2, Kilian 2026-07-24: LLM supplies the enemy model SACRED provably cannot infer - the gen34 rescue)

**status: PRE-REGISTERED 2026-07-24.** Mandate: Kilian's 2026-07-24 direction (approved
in-conversation): an LLM experiment that genuinely ASSISTS SACRED and is relevant. This act
gives the LLM the one job it is built for (natural-language narrative -> classification) at the
exact point where gen34 measured SACRED failing: inferring the hidden enemy TYPE. Full launch
control granted. Ledger committed BEFORE the harness runs; results appended below the RESULTS
line; nothing above it changes.

**git SHA:** the commit landing this ledger; steps pin their own SHAs.

## Why here (and why this one can win where curation could not)

The failure pattern across B2 (knows concepts, cannot produce numbers), gen33 (no terrain
grounding) and gen37 (worse-than-random route curator) is consistent: **the LLM loses whenever
asked to do quantitative/combinatorial work.** gen38 asks it instead to do LANGUAGE ->
DECISION, its native strength, at a point where it can supply an ingredient SACRED provably
lacks.

gen34 (2026-07-24) built the exact apparatus and measured the wall: the enemy is secretly one
of five doctrines; knowing which is worth 1.39-2.04x on the held-out cells (the inference gap);
the type-BLIND SACRED generalist could not cross the blind cap (pooled 1.373x it, 0/18 cells),
and the "know-the-playbook" ceiling (omni: play each type's exact specialist) captures the full
gap. gen38 tests whether an LLM, reading a behavioural INTELLIGENCE ASSESSMENT of the enemy
(worded in how-they-act terms, type never named), can identify the doctrine well enough that
deploying the matching counter crosses the banked wall. This is "LLM assists SACRED": the model
contributes the enemy model the learning system could not infer from raw observations.

## Inherited exact apparatus (gen34, `models/runs/gen34_hidden_adversary/family_refs.json`;
## specialists via `scratch/gen34_family_probe.py` machinery + `scratch/dyn_exact.py`)

Per held-out instance (6 Gdansk cells): the five members' cost matrices, each member's exact
optimal specialist policy (`greedy_policy_from_rvi`), the per-type omni values, the omni_cap
(mean = perfect-type-ID value), and the blind_cap (best type-blind policy = THE WALL). The
brittleness cross-table (specialist[i] vs enemy[j]) is the reported risk of a wrong call.

## The act (pinned)

**Step V1 (oracle-level, LLM classification; the decisive cheap test, launched now):**
- **Narratives:** 20 hand-authored intelligence assessments (4 per member), behavioural, the
  type NEVER named, committed verbatim in `scratch/gen38_narratives.py`. Authored before any
  LLM call; disclosed as author-written (the anti-circularity control is the keyword baseline +
  the confusion matrix, not blind authorship).
- **LLM classifier (llama-3.3-70b via the tunnel):** shown the five doctrine descriptions
  (the analyst's known playbook) + one narrative; returns {type, confidence 0-1, reasoning} as
  JSON; temperature 0.2; 3 independent draws per narrative (consistency). Full transcripts
  committed.
- **Controls:** (a) naive KEYWORD classifier (fixed keyword->type table, committed); (b) RANDOM
  classifier (uniform 5-way, analytic expectation).
- **Operational eval (oracle-exact):** the LLM-assisted defender plays specialist[predicted]
  against the true type. Per held-out instance, value = mean over the 20 narratives of
  `policy_value_exact(specialist[pred], cost[true])`. Pooled over 6 cells. Also the
  CONFIDENCE-GATED variant: if the LLM confidence < 0.5, play the blind policy (hedge) instead
  of committing to a specialist - reported beside the commit-to-argmax primary.

**Step V2 (gated on V1 PASS; the full "SACRED assisted by LLM" confirmation):** train ONE
type-CONDITIONED SACRED policy (the type index appended to the head features; an easier task
than blind inference), 3 seeds, then deploy it with the LLM-supplied type at eval. Shows the
trained agent + LLM crosses the blind cap that the trained agent alone (gen34) could not. Same
gen34 pool/config; NEW flag `--type-conditioned` in `scripts/train_family_generalist.py`
(additive; flag-off byte-identical; suite green at the build record).

## Decision metric (PRE-REGISTERED)

Operational value = pooled held-out expected per-sortie loss of the LLM-assisted defender;
ratio to the blind_cap (the wall). Classification accuracy = fraction of the 60 LLM draws whose
argmax type is correct.

> **PRIMARY (crosses the wall): the LLM-assisted (commit-to-argmax) pooled operational value <
> blind_cap on >= 4/6 held-out cells.** No type-blind object can do this by construction; the
> trained gen34 generalist did not.
> **STRONG: pooled operational value <= omni_cap x 1.15** (within 15% of the perfect-type-ID
> ceiling), i.e. the LLM captures most of the inference gap.
> **COMPARATIVE (reasoning is the ingredient): LLM classification accuracy > keyword-baseline
> accuracy AND LLM operational value < keyword operational value.** An LLM ~ keyword result is
> "lookup suffices", reported as such.
> **MECHANISM (reported): the 5x5 confusion matrix (which doctrines the LLM confuses), the
> confidence calibration, and the commit-vs-hedge delta.**
> **Branches (all writable):** PASS (primary+comparative) = the headline: *an LLM reading a
> plain-language intelligence assessment supplies the enemy doctrine that SACRED could not
> infer from observations, crossing the exact type-blind wall gen34 banked; naive keyword
> lookup does not suffice.* V2 then confirms the trained agent realises it. PARTIAL = crosses
> on some cells / needs confidence-gating / accuracy good but operational value hurt by
> brittle wrong calls (the cross-table risk) = report per-clause with the hedge. FAIL = the LLM
> cannot classify the doctrine from behaviour, or its wrong calls cost more than blindness -
> the strongest negative, completing the arc that the LLM helps SACRED NOWHERE in this pipeline
> (B2/gen33/gen37/gen38), a clean and publishable boundary on LLM-for-security-games.

## Design decisions ledgered

1. V1 oracle-level first (LLM calls + exact policy values, no training): the LLM-classification
   question is decisive and cheap; V2 training is not spent until V1 shows the type signal is
   there (the gen37 discipline: cheap diagnostic gates the training spend).
2. Specialists are the gen34 EXACT optimal per-type policies in V1 (isolates the LLM question
   from any training gap); V2 swaps in the trained type-conditioned SACRED for the full claim.
3. Narratives author-written and committed verbatim; the anti-circularity guard is the keyword
   control + confusion matrix, plus a pre-registered rule: NO narrative is edited after seeing
   any LLM output.
4. Confidence-gating reported, not primary: committing to the argmax is the honest hard test;
   hedging is the deployable refinement.
5. Commit-to-argmax uses each draw independently (60 draws); majority-vote-over-3 reported as a
   variance-reduced row.
6. Numbers live only in this ledger + its JSONs; transcripts committed under
   `models/runs/gen38_llm_enemy_id/transcripts/`.
7. Thread caps per SYSTEM.md on any V2 launch.

## Commands (pinned)

```bash
# V1 (LLM classification + oracle operational eval; launched now):
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen38_enemy_id.py
# V2 (gated on V1 PASS): type-conditioned SACRED, 3 seeds, then LLM-supplied-type eval.
```

## Compute envelope

V1: 60 LLM calls (~15-40 s each) + oracle policy-values (seconds/instance) = well under an
hour. V2 (if gated in): one gen34-scale training batch (3 seeds ~ overnight) + oracle eval.
No extension without a dated amendment before results.

## RESULTS (appended per step; nothing above changes after launch)

### RESULT V1 (2026-07-24; oracle+LLM, NO training; `scratch/gen38_enemy_id.py`,
### `models/runs/gen38_llm_enemy_id/{v1_result.json,transcripts/}`; ~60 LLM calls)

**Apparatus reproduced gen34's banked held-cell caps exactly** (e.g. 249-95 blind_cap 0.1198 /
omni_cap 0.0589) before any new number was read.

| quantity | value | note |
|---|---|---|
| LLM classification accuracy | **1.000** (60/60 draws; perfect diagonal confusion) | |
| keyword-control accuracy | 0.800 | strong baseline, but its 20% errors are operationally CATASTROPHIC |
| random-control accuracy | 0.200 | |
| blind_cap (THE WALL) | 0.1140 pooled | best type-blind play; gen34's trained generalist reached only 1.373x it |
| omni_cap (perfect type-ID) | 0.0631 pooled | the ceiling |
| **LLM commit-to-argmax** | **0.0631 pooled = omni_cap; crosses the wall 6/6 cells** | 100% ID -> always the right specialist |
| keyword commit | 0.1942 pooled; crosses 0/6 | WORSE THAN BLINDNESS: wrong calls deploy brittle wrong counters (the cross-table biting) |
| random commit | 0.2925 pooled | |

> **VERDICT V1: PASS on every clause. PRIMARY 6/6 (bar >=4/6); STRONG (<=1.15x omni_cap);
> COMPARATIVE (LLM acc 1.0 > kw 0.8 AND LLM value 0.063 < kw 0.194).** The first genuine
> positive in the LLM strand: an LLM reading a plain-language intelligence assessment
> identifies the enemy doctrine and supplies the type the type-blind SACRED generalist
> provably could not infer (gen34), crossing the exact banked wall. The comparative is sharp -
> naive keyword lookup, though 80% accurate, CROSSES 0/6 because its confident wrong calls are
> worse than blindness (mean 0.194 > blind 0.114): correct doctrine ID, not surface matching,
> is the ingredient. This is language->decision, the LLM's native register, and it succeeds
> exactly where the quantitative registers (B2 numbers, gen33 grounding, gen37 curation)
> failed - the honest shape of "where LLMs help this pipeline".
>
> **Scope + skeptic's caveats (binding):** (1) V1's counter is the ORACLE specialist; the full
> "trained SACRED assisted" claim needs V2 (type-conditioned SACRED), now GATED-IN and
> launched. (2) The 20 narratives are author-written and behaviourally clear; 100% may be
> optimistic. A ROBUSTNESS row (perturbed/noisier/mixed-cue narratives) is run ungated below to
> probe fragility - the operational value rests entirely on classification accuracy, so its
> robustness IS the result.

### ROBUSTNESS ROW (2026-07-24, ungated/disclosed; `scratch/gen38_robustness.py`,
### `models/runs/gen38_llm_enemy_id/robustness.json`; the skeptic's attack on the clean 100%)

Narratives degraded PROGRAMMATICALLY (not re-authored): TERSE = first sentence only;
DISTRACTOR = full assessment + one CONFLICTING sentence from a different doctrine (realistic
contradictory intel); BOTH = terse + distractor. Blind_cap 0.1140, omni_cap 0.0631.

| condition | LLM acc | keyword acc | LLM op value | keyword op value | LLM crosses wall |
|---|---|---|---|---|---|
| clean (V1) | 1.000 | 0.800 | 0.0631 | 0.1942 | 6/6 |
| terse | 0.950 | 0.800 | 0.1035 | 0.1758 | 5/6 |
| **distractor** | **0.800** | 0.450 | **0.0895** | 0.3701 | **6/6** |
| both (terse+distractor) | 0.400 | 0.400 | 0.2175 | 0.3659 | 0/6 |

> **Reading (binding):** the result is ROBUST to realistic messy intel and the reasoning-vs-
> lookup gap WIDENS under it. The distractor condition is the decisive discriminator - with a
> contradictory sentence spliced in, surface matching collapses (keyword 0.80 -> 0.45, op
> 0.19 -> 0.37) while the LLM holds 0.80 and still crosses the wall 6/6: correct doctrine
> reasoning, not pattern-matching, is confirmed as the ingredient. The FRAGILITY BOUNDARY is
> honest and disclosed: when intel is BOTH minimal AND self-contradictory ("both", 0.40 acc)
> classification fails for everything and commit-to-argmax is worse than blindness (0/6) - this
> is exactly the regime for CONFIDENCE-HEDGED deployment (play the blind policy when the model
> is unsure), the pre-registered hedge variant, which is the deployment recommendation. The V1
> headline stands with the scope: *the LLM supplies the enemy doctrine and crosses the gen34
> wall under clean, terse, or contradictory intelligence; it should hedge to the blind policy
> when the assessment is both sparse and conflicting.*

### RESULT V2 (2026-07-25 00:35; batch `scratch/gen38_v2_batch.sh` at SHA `475698b`;
### artefacts `models/runs/gen38_llm_enemy_id/v2_seed{0,1,2}.{json,log}` + ckpts; 3 seeds x
### 12000 sorties. One disclosed mid-act repair BEFORE any result was read: the first V2
### launch's type one-hot was INERT (constant across routes cancels in the softmax; policy sat
### at 1.38x blind = gen34's blind level); replaced by the per-route type-threat column
### (commit `475698b`), verified type-discriminating offline; the aborted run's outputs were
### discarded unread beyond the diagnosis)

| seed | best told-TRUE-type (ratio-to-blind-cap) | told-LLM-type at that checkpoint |
|---|---|---|
| 0 | 0.670 @ 8000 | 0.673 |
| 1 | 0.664 @ 7000 | 0.663 |
| 2 | 0.657 @ 12000 | 0.660 |
| pooled | **0.664** | **0.665** |

> **VERDICT V2: CONFIRMED 3/3 seeds - the full "SACRED enhanced via LLM reasoning" claim.**
> The trained type-conditioned policy crosses the type-blind wall decisively (pooled 0.664x
> the blind cap, vs the gen34 blind generalist's 1.373x - a 2.07x improvement), capturing
> ~75% of the exact inference gap ((blind 0.1140 - achieved 0.0758) / (blind - omni 0.0631)),
> and **the LLM-supplied type is operationally indistinguishable from the truth** (pooled
> delta 0.001-0.003 across seeds; V1's 100% classification carries through end-to-end). The
> conditioning-capacity lesson recurs and is disclosed: the type signal only works delivered
> as a per-route DISCRIMINATING feature (the threat column), not as a symbol (the inert
> one-hot) - the same head limitation gen36/gen34 measured, here engineered around.
>
> **The banked act-level claim (V1+robustness+V2):** *an LLM reading a plain-language,
> possibly self-contradictory intelligence assessment identifies the enemy doctrine (100%
> clean, 80% under contradictory intel where keyword lookup collapses to 45% and is worse
> than blindness) and hands it to a trained SACRED policy, which then crosses the exact
> type-blind performance wall it provably could not cross alone (0.66x vs 1.37x the blind
> cap), capturing ~75% of the theoretical value of knowing the enemy; deployment should
> confidence-hedge when intel is both sparse and contradictory.* This is the LLM strand's
> positive, in the language->decision register, completing the measured arc with B2/gen33/
> gen37's quantitative-register negatives.
