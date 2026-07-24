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
