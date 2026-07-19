# Generation: gen31_aerial_dyn (the gen27 conversion: an aerial dynamic register with a real corridor)

- **status: PRE-REGISTERED 2026-07-19 (Kilian's in-conversation mandate: reopen the aerial
  trained register, iterate until aerial SACRED shows a gen27-comparable positive; unlimited
  budget; standing autonomous launch authority FOR THIS ACT; full enemy-design freedom, biased
  towards SACRED winning; calendar unconstrained). This supersedes the v4.0-dyn closure's
  "do not reopen" for this NEW register only; the v4.0-dyn negative and its corridor-collapse
  finding stand as measured history and are the design input here.**
- **branch:** `gen28-aerial` (worktree `../sacred-aerial`), additive files only; ledger first.
- **git SHA:** the commit landing this ledger; every attempt pins its own.

## The mission (what "gen27-comparable" means, agreed with Kilian)

gen27's surviving claim shape, transplanted: **one history-aware fleet policy, trained across
aerial layouts, evaluated ZERO-SHOT on held-out layouts, beats every STATIC object (the
iid_eq cap and the local static optimum) on >= 4/6 held-out layouts AND pooled, on >= 2/3
seeds, with a BLINDED (no-window) causal control landing at the cap.** A hand-composed or
oracle-fitted dynamic rule remaining somewhat ahead does NOT fail the act (gen27's composed
rule was ahead and the act still banked); the wording clears exactly the rows the numbers
clear. Payoff-blind dynamic rules (anti-repeat/rotation families) should also be beaten for
the result to be interesting; that is the aiming target of the corridor hunt, not a hard bar.

## Why the literal transplant died, and what this act changes (the gen27 anatomy)

gen27 worked because five conditions held: (1) a guaranteed corridor (every static object
capped); (2) the optimal behaviour was EXPRESSIBLE by the policy's window channel; (3) no
cheap rule sat on the optimum; (4) proven trainability plumbing (N=3 fleet menu-select,
per-instance smooth FP, undiluted head features, validation selection); (5) a zero-shot
transfer register. v4.0-dyn (the literal transplant) failed conditions 2 and 3 jointly on the
aerial game: with recency-only information the only expressible behaviour is anti-repeat
(provably bad on structured layouts); with doctrine information a two-line myopic dodge is
strong. **gen31 redesigns the ENEMY and the INFORMATION CHANNEL until conditions 1-3 hold,
verified oracle-exactly, before any training CPU.** Plumbing (4) and register (5) reuse the
committed v3.1 fleet machinery unchanged.

## The design axes (full freedom, recorded; all keep the window-MDP exact at w <= 2)

1. **Anticipatory mixed doctrine (the lead lever).** The enemy's aim distribution is a
   softmax(tau) over per-position expected damage against a MIXTURE of defender models:
   with weight q_rep "they repeat their pattern" (damage vs the realised window, the v4.0
   doctrine), with weight q_dodge "they will dodge their pattern" (damage vs a dodger who
   plays uniformly over routes NOT in the window), with weight q_eq a committing
   equilibrium-attacker component. Militarily this is an enemy that hedges between pattern
   exploitation and evasion anticipation. Design intent: pure repetition dies (q_rep), pure
   myopic dodging dies (q_dodge: the enemy is waiting where you flee), pure static hedging
   dies (adaptivity), so optimal play is a window-conditioned CALIBRATED mixture: exactly
   what a policy can express and a two-line rule cannot.
2. **Information channel:** the policy head sees, per route: exposure (static), window
   recency, and the DOCTRINE column (this sortie's expected damage per route given the
   window). Information parity is binding: every rule in the family gets the same columns.
3. **Operating point:** w in {2, 3}, tau in {0.10, 0.15, 0.25}, (q_rep, q_dodge, q_eq)
   simplex points; layout family (structured double-pinch primary, open-banded context);
   K=1 first (K=2 exact matrices move to w05, recorded).

## The corridor-hunt gate (Phase 0, oracle-only, FREE; no training below these gates)

Per candidate operating point, on >= 3 probe layouts, computed exactly (RVI with the
aperiodicity transform; stationary rule values by the exact chain machinery):

- **G1 (static corridor):** iid_eq / history_opt >= ~1.4 AND the multi-start local static
  optimum stays within a few % of iid_eq (static play genuinely capped).
- **G2 (payoff-blind dynamic rules beatable):** min over the anti-repeat/rotation family
  (every lane spacing + full menu + eq-support variants) >= ~1.25x history_opt.
- **G3 (fitted-rule context):** the doctrine-informed myopic dodge, a temperature-fitted
  softened dodge, and hedge-composed variants (all oracle-fitted, disclosed as caps) are
  computed and recorded; the corridor of interest is what remains below the BLIND family
  even if a fitted rule is close to the optimum (the gen27 composed-rule precedent).
- **G4 (representability):** a small trainable function of the pinned feature columns
  (softmax head over [exposure, recency, doctrine, static-hedge] with few parameters,
  fitted by direct stationary-value search) reaches materially below the payoff-blind rule
  family towards history_opt: proof the policy CLASS can express the win before deep RL
  attempts it (the formalised v4.0 lesson).
- **G5 (standing non-degeneracy):** trainable asymmetry; values inside (0.02, 0.9).

If NO operating point in the whole design space passes G1-G5, that result goes to Kilian
before any training (pre-written fail branch; the v4.0 finding would then be confirmed at
full generality).

## The iteration protocol (binding; Kilian's iterate-until-done mandate, made safe)

- Iterate freely on TRAIN and VALIDATION layouts: any number of attempts, each attempt gets
  a dated ledger amendment with config, result, and a mechanism autopsy BEFORE the next
  change. No silent re-rolls.
- **The 6 GATED held-out layouts and the confirmation seeds are never touched during
  iteration.** When a configuration passes on validation, it runs ONCE, blind, on the gated
  set with fresh seeds; that confirmatory run is the citable result. If confirmation fails,
  iteration may resume but the failed confirmation is disclosed and a NEW gated set is drawn
  for the next confirmation (no gated set is ever reused after being seen).
- Bars at confirmation (pinned now, gen27's shape): **PRIMARY: zero-shot per-layout
  stationary damage < that layout's iid_eq cap on >= 4/6 AND pooled, on >= 2/3 seeds, at the
  validation-selected checkpoint. CAUSAL: the blinded (window-zeroed) arm lands ~ the cap.
  STRONG: pooled <= 2.5x history_opt.** Reported ungated: the full rule-family ladder,
  worst-case-vs-committing row, final-iterate drift, per-layout values.
- Trainer = the committed v3.1 fleet generalist (N=3, menu-select, per-instance smooth FP,
  exposure+recency+doctrine head columns at dedicated lr, validation checkpoint selection);
  thread caps + nice as standing. M4 first; w05 for breadth (parallel design arms) and any
  K=2 cell, with Kilian informing the professor before sustained use.

## RESULTS / ITERATION LOG (appended per attempt; nothing above changes)
