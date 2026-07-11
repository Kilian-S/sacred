# NEXT_STEPS_11-07-26: suggested further work to the freeze (agreed Kilian + critic instance, 2026-07-11)

> **Provenance and status.** This is the amended pre-freeze plan negotiated between Kilian and the
> 2026-07-11 critic instance (author of `CRITIQUE_EXPANSION.md`), written as a SUGGESTION of
> further work for the implementing instance. It is not yet launched; Kilian's answers to the
> opening questions in §6 activate it. Calendar: Kilian has confirmed ~2 weeks of experimental
> runway; the Final Activities Report is due 30 July; freeze 3 August HARD; thesis + poster
> 10:00, 28 August. The B section (B1-lite-2, full B1, B3, B5) is frozen by agreement.
>
> **Read first:** `HANDOVER.md` top banner; `CRITIQUE_EXPANSION.md` (the critique this plan
> derives from: §4.1 the ZST-vs-LP framing, §4.2 the measured dual-selection numbers, §4.3 the
> gen19 framing, §7 the performance items, §9 the original programme); `experiments/
> gen19_b1lite1.md` (the latest banked result, PASS+STRONG).

## 0. Mission and spirit

The overarching goal is that SACRED shows positive results and is successful, achieved the way
this project always has been: favourable-but-honest problem selection via pre-registered screens,
metrics fixed before looking, instances pinned by probes never by outcomes, failures reported
with mechanism. A pre-registered failure with a mechanism is a reportable result; a rescued
number is not. The project's credibility is its honesty; every item below is designed to survive
a hostile examiner, not merely to add a row.

**Operating rules (hard, unchanged):** every generation gets its own pre-registered ledger
(question, metric, gates, pinned SHA) committed BEFORE any training; citable numbers live only in
ledgers; oracle/screen probes are free; run the suite after any `src/` change and paste the raw
output; all new machinery additive and flag-gated (absent flags = byte-identical paths);
best-checkpoint discipline with drift disclosed; TAP estimators; **select best checkpoints on
TRAIN-side metrics and report held-out at that checkpoint** (the §4.2 lesson, the new default);
never compare across git states; long runs via detached orchestrators with JSON + log + per-eval
checkpoints saved; append a `SACRED_PROGRESS.md` entry as each run family completes; commit as
you go.

**Closed doors (not to be reopened under any outcome):** last-iterate convergence (the gen17
gate), leader stabilisation (gen09-STAB), learned-follower coordination (the gen18 gate), the
whole B section, further leader experimentation, and 62-97 as a headline instance.

**Concurrency note:** a second agent may be building a mission-control application in a SEPARATE
directory that reads this repo as read-only data. The implementer should never write outside this
repo, should not be surprised by that agent's existence, and owes it nothing beyond the existing
commit discipline (ledgers/JSONs/logs are its data feed).

## 1. The suggested work, in the agreed order

### Item 0: chronicle + doc hygiene (first; it blocks the 30 July FAR)

Append `SACRED_PROGRESS.md` entries for gen13 through gen19 and the expansion ledgers
(a2/a3/a4/b4/d1/d2/d3/f3); fix the `HANDOVER.md` "entries 17-19" pointer and refresh the top
banner; consolidate stale banner layers where cheap. Draft the FAR skeleton and presentation
outline per Kilian's answer to question 4 (§6). Zero CPU; can run while the opening questions are
pending.

### Item 1: F2 (gen20): ONE clean learned-interdictor attempt, post-fix

A learned antagonist replaces the oracle BR as the SPARRING PARTNER only; EVALUATION stays
oracle-BR portfolio-max in every row (a weak learned attacker must not be able to flatter the
defender). Pre-register the bar before launch; suggested shape: the learned-adversary-trained
defender's best-checkpoint TAP lands within a stated margin of the oracle-trained reference on
the same instance; FAIL = the honest oracle-bounded sentence for Obj-1, with the learned
attacker's strength relative to the oracle BR reported. **HARD GATE: one attempt, no chase,
whatever the outcome.** This is the third critique in a row recommending it; it closes the last
gap against Obj-1's verbatim wording ("environment-altering antagonist agent": currently an
oracle in every positive result).

### Item 2: the ZST hardening batch (gen21)

1. **Vanilla-generalist control:** the gen16 config with the travel objective and no adversary,
   1 seed; evaluate zero-shot on the same held-out Gdansk ODs. The missing Obj-5 control at the
   transfer level ("adversarial training is what makes transfer work" becomes measured); either
   outcome is citable.
2. **Dual-selection report:** re-verify `CRITIQUE_EXPANSION.md` §4.2 from the saved JSONs (gen15
   1.592 identical under both selections; gen16 1.677 test-selected vs 1.733 train-selected) and
   fold both numbers into the gen15/gen16 ledgers as disclosed rows.
3. **Hold-out-Istanbul rotation cell:** the gen16 recipe with training on Kaliningrad + East
   London + Gdansk and Istanbul held out entirely, 3 seeds, the same pre-registered bars as
   gen16. (The single most informative rotation cell: transfer to the structurally most distant
   city; the full leave-one-city-out rotation is deliberately NOT funded.)
4. **Zero-shot K/N rows (eval-only):** the frozen gen16 actors evaluated at K=2 and N=5 on
   held-out ODs against each cell's own oracle anchors.
5. **Whole-Kyiv zero-shot row (CONDITIONAL on Kilian providing the OSM export, question 1):**
   extract with `scripts/extract_city.py`, length-repair, oracle-screen sampled ODs (free),
   pre-register gen16-style bars, then evaluate the frozen actors zero-shot. Eval-only; claim
   wording is "the whole city's arterial network, same pipeline"; if it fails it is the measured
   boundary of the scale axis, reportable. If the export has not arrived by mid-plan, drop this
   row and let 2.3 carry the axis.

### Item 3: C1 (gen22): ERB bootstrapping via a population-based metaheuristic, done literally

Seed the replay buffer with ALNS-plan demonstration transitions on 35-159; arms {seeded, cold}
x 3 seeds; pre-registered time-to-competence primary (e.g. sorties to first reach a stated TAP
bar, plus final best-checkpoint parity). Closes Obj-3's verbatim wording either way.

### Item 4: A4 large-K (gen23), in four bounded steps

1. Wire `greedy_br_attacker` into the trainer's attacker refresh and the exploitability eval;
   gate the eager objective matrix behind K <= 3; regression test; suite green.
2. **Fidelity gate at K=3** (exact still computable): the greedy-BR-trained ladder must reproduce
   the exact-pipeline ladder within a pre-registered tolerance. No pass, no step 3.
3. **The K=5 cell on 35-159** (exact enumeration infeasible: ~22.5M interdiction sets, ~70 GB):
   all arms (shortest-path stack, ALNS, SACRED) scored under the SAME greedy yardstick;
   exploitability reported as the certified interval **[v_greedy, v_greedy / (1 - 1/e)]**.
4. The infeasibility figure: exact solve cost vs K up to the wall, with the trained K=5 point
   past it. Concede column generation in one sentence, per the standing position.

The situations this evidences, concretely: multi-asset adversaries (the blow-up is the ENEMY's
combinatorics); large theatres (candidate-edge count multiplies the same wall); instance streams
where per-solve cost stops being milliseconds; and design loops (D3) under a capable adversary,
where the trained-policy pipeline is the only runnable one.

### Item 5: the D3-on-Gdansk composite exhibit (eval-only)

The D1 acquisition loop over placement x fleet designs ON THE HELD-OUT CITY, priced by the frozen
multi-city generalist (one forward pass + one BR per design). Strategic design in a never-trained
theatre: the poster exhibit.

### Item 6: gen19 dress-up (cheap)

Fold the (w, tau) screen grid into the gen19 ledger as sensitivity curves if not already recorded.

**Drop order if anything overruns despite the runway:** 5, then 2.3, then 6, then 4.3-4.4 (keep
4.1-4.2), then 3. Items 0, 1 and 2.1-2.2 survive any schedule.

## 2. The storyline review phase (mandatory, after the experiments or when the calendar demands)

1. Rewrite `THESIS_STORYLINE.md` onto the four-act spine (`CRITIQUE_EXPANSION.md` §4.7): the
   negative campaign compressed; the security game with the two post-fix ladders; ZST + the SBO
   stack as the payoff; the measured boundaries (transients, structural stacking,
   identity-vs-semantics, gen19's bounded-adversary dynamism) as the discussion. Fold in every
   new result from this plan. Write the ZST-vs-LP paragraph (§4.1) and the gen19 quantal-response
   framing (§4.3) explicitly; score the five objectives honestly, deltas named, no "all
   objectives met" shorthand.
2. Then the self-critique: interrogate the rewritten storyline as a hostile examiner, thoroughly,
   comprehensively and objectively: triviality, logical gaps, overclaims, undisclosed scope
   conditions, missing controls, statistical wording, framing drift, SDVRP-title honesty, ALNS
   wording, related-work positioning. Write the findings as a dated, numbered list.
3. Then CORRECT the findings autonomously: edit the docs; add cheap eval-only rows or probes
   where a finding needs one; re-verify any number touched against its ledger. Anything needing
   non-trivial new training beyond this plan is recorded as a proposal for Kilian, not launched.
   Iterate the critique once after corrections; stop at two rounds.
4. Finish with: refreshed `HANDOVER.md` top banner, updated memory, everything committed, and a
   wrap report to Kilian (what ran, what passed/failed, what changed in the storyline, what
   remains).

## 3. Decision rule when a choice arises and Kilian is unavailable

In priority order: (1) the five research objectives verbatim and the aim (lit review §2.2, the
PDF at `../../MT_Literature_Survey_Kilian_Schwarz_split.pdf`); (2) recorded design decisions,
dogmas and exit criteria (`SYSTEM.md`, `HANDOVER.md`, the ledgers, the four critiques); (3) the
overarching goal that SACRED demonstrably succeeds, through honest means; (4) between ties, the
option that yields a citable pre-registered result sooner. Never patch a metric after looking;
never reopen a closed door; a decision that is genuinely Kilian's (new scope, big CPU, anything
outward-facing) is queued as a question while work continues on other items.

## 4. Launch authority

Suggested arrangement (to be confirmed by question 2): batch launch authority for items 0-6
exactly as written, granted once at activation, so no per-run go is needed; anything OUTSIDE this
plan's scope keeps the standing rule (Kilian's explicit go). Standing etiquette unchanged: no
multiple-choice prompts (prose + firm recommendation); single &&-chained commands when giving
Kilian shell lines; he pauses runs for heat/noise.

## 5. Not funded (for the record, so it is a decision rather than an omission)

The full leave-one-city-out rotation (one cell funded instead, 2.3); B1-lite-2 / full B1 / B3 /
B5 (B section frozen); any further leader, coordination or last-iterate work; wall-clock scaling
claims at small instance sizes (the A3 position stands).

## 6. Opening questions for Kilian (the implementer asks all in one message, then begins)

1. **Kyiv:** will you provide the OSM export (geojson), and roughly when? If it will not arrive
   within the plan window, row 2.5 drops and the Istanbul rotation carries the axis.
2. **Authority:** do you confirm batch launch authority for items 0-6 as written (any per-item
   vetoes or additions)?
3. **Compute etiquette this fortnight:** preferred training hours, 3-parallel cap, heat/noise
   pause rules, and whether scheduled wakeups are permitted.
4. **FAR:** shall the implementer draft the Final Activities Report (2 pages) and the
   presentation outline as part of item 0, or will you write those yourself from the refreshed
   chronicle?
5. **F2 instance:** any preference (35-159 multi-convoy vs 33-71 single-convoy), or the
   implementer's call with justification in the ledger?
6. **Slack:** with ~2 weeks confirmed, anything you want added to the plan now rather than after
   the storyline review?
