# ROADMAP.md: the active plan (opened 2026-07-06)

> **This supersedes `TASK.md` as the active plan** (TASK.md banner points here). Strategy and
> rationale: `DIRECTION.md`. Narrative: `THESIS_STORYLINE.md`.
>
> **Standing rule (Kilian, 2026-07-06): plan first, never dive in.** Every phase below opens with
> a written plan (this file + the relevant ledger) and closes with a recorded result. Items
> marked **⛔K** need Kilian's explicit go (CPU spend, launches, design changes, external
> communication); **⛔S** additionally need supervisor agreement. When unsure, consult Kilian.
> Hard calendar rails: thesis + poster due **10:00, Fri 28 Aug 2026** (12,000 words max);
> experimental freeze **Aug 3, HARD (Kilian 2026-07-06)**; after Aug 3, writing wins every
> conflict.

## Phase A: sign-off and zero-CPU groundwork (SHORT TERM: Jul 6-12)

- [x] **A1. Decision agenda: RESOLVED BY KILIAN 2026-07-06** (see DIRECTION.md §9 log):
      (a) reframe ADOPTED; (b) freeze Aug 3 hard; (c) Obj-4 reduced form; (d) rolling-ALNS arm
      funded; (e) ERB ablation included, modest scope; (f) ZST one transfer test. The
      supervisor conversation itself still happens (Kilian's comms; framing question from
      DIRECTION decision 2), but the build no longer waits on it: Kilian authorised the
      Phase-B start 2026-07-06. (Conference/publication topics remain OFF the table.)
- [ ] **A2. Thesis planner launch** (⛔K: Kilian opens the session in `../../thesis/` and says
      "read THESIS_PLANNER_HANDOFF.md and begin"). Pre-step DONE 2026-07-06: the handoff file
      carries the redirection banner (trio pointers, three-act branch-robust structure, updated
      ledger list and timeline). **Kilian 2026-07-06: not now** (launch timing stays his call).
- [x] **A3. Evidence-hardening probes**: **DONE 2026-07-06 (overnight)**; approved by Kilian,
      executed autonomously; results appended to the gen05/gen06 ledgers as post-hoc analyses
      (primaries untouched). Headlines: A3.1 telemetry reproduced and committed; A3.2 vanilla's
      aimed-attack robustness DECLINES with training (specialisation → predictability: direct
      in-house support for the exploitability register); A3.3 the gen06 gap is NOT sampling
      temperature (persists/widens at matched determinism; tau 1.0 sanity rows reproduce the
      ledger exactly); A3.4 dual-level statistics recorded (pooled significant + 3/3 signs;
      3-pairing t-CI includes zero for gen06, excludes it for gen05). Scripts:
      `scratch/gen06_telemetry_probe.py`, `scratch/gen06_snapshot_robustness.py`,
      `scratch/gen06_matched_temperature.py`, `scratch/gen0506_seedlevel_stats.py` (+ JSONs).
      Original sub-item specs (for the record):
      - A3.1 `scratch/gen06_telemetry_probe.py`: reproduce and commit the arm-comparison
        telemetry (alpha, entropy, queue, delivery, Q_Spread, critic loss) currently recorded
        only as the session analysis in `DIRECTION.md` §4. Zero CPU beyond seconds.
      - A3.2 Robustness-vs-training-time: re-evaluate all gen06 snapshots under
        pathrand/targeted on validation instances (~16 snapshots x 6 runs x 8 instances,
        eval-only, ~minutes-to-an-hour). Explains the ep100 vanilla selections; likely a thesis
        figure. ⛔K.
      - A3.3 Matched-temperature diagnostic: evaluate gen06 selected checkpoints at matched
        determinism levels (both arms sharpened equally; labelled diagnostic, dogma-compliant).
        Distinguishes "knowledge deficit" from "temperature deficit". Eval-only, ~30-60 min. ⛔K.
      - A3.4 Seed-level statistics note for gen05/gen06 (dual-reporting rule). Zero CPU.
- [x] **A4. gen07 ledger draft**: **DONE 2026-07-06** (Kilian confirmed after the morning
      clarification): `experiments/gen07_contested_matrix.md` opened as a DRAFT
      pre-registration (commit `2089e1f`): portfolio-max exploitability estimator with paired
      bootstrap + dual-level stats; arms vanilla/dr/sacred + eval-time entropy-matched control;
      five pre-launch gates (suite, timing, competence/recoverability, BR gate, coping-channel
      probe) with a pre-registered arena escalation rule; five interpretive branches. **Nothing
      runs from it until Phase B gates pass and Kilian launches** (⛔K); TO-FINALISE slots may
      be pinned only by probes, never by outcomes.

**Exit criteria:** A1 decisions recorded in `DIRECTION.md` §9; A4 ledger reviewed by Kilian.
**If A1 rejects the reframe:** fall back to the recorded freeze-and-write on gen06 (still fully
defensible); this file gets a closure banner; A3 outputs remain valuable for the thesis either
way.

## Phase B: build the five fixes (MID TERM: ~Jul 13-18, ~3-4 focused days)

All behind flags, additive, suite-guarded (`PYTHONPATH=. pytest tests/` after each item, raw
output pasted). No behaviour change to any historical mode (gen03-06 configs must reproduce).

> **BUILD PROGRESS (2026-07-06 overnight, branch `gen07-contested`).** The five fixes + arena are
> BUILT, tested and committed; Kilian chose B1 Option B and B4-lite. Done:
> **B6** contested arena (`15fd798`), **B2+B5** entropy repair + gamma (`9557ced`), **B1** twin
> difference reward Option B (`a5a818e`, invariant verified numerically), **B3** exposure/strength
> curriculum (`c7e0ba6`), **B4-lite** scripted-attacker population (`6e9bffd`), **B7** contested
> ERB generator (`96d96c0`). Suite **107 green**; every historical mode preserved (all new
> behaviour flag-gated, regression-guarded); the FULL gen07 sacred stack (twin + curriculum +
> mixture + gamma 0.997 + absolute entropy targets) smoke-trains end-to-end in the viable band
> (delivery ~65-68%, not the gen06 collapse regime). Training NOT launched (Kilian greenlights).
> **Deferred: B8** rolling-ALNS (eval-only, blocks nothing; carries a faithfulness/scoping
> judgment for Kilian, see B8 below). The factored learned-antagonist head (B4 part i) is deferred
> with B4-full/the BR gate (B4-lite uses scripted attackers, so it does not need it).

**Separation policy (Kilian 2026-07-06: "make sure the changes are not deleterious to what we
have already built"):** three layers. (1) `main` stays the frozen campaign record: from
2026-07-06 onward no `src/` changes land on `main`; docs, ledger appendices and `scratch/`
analyses may. (2) All Phase-B/C code work happens on a dedicated branch (`gen07-contested`,
created when Phase B starts) that merges only after the suite is green and Kilian reviews.
(3) Within the code, every gen07 behaviour is behind a new flag with defaults preserving the
historical modes, enforced by regression tests. Ledgers pin SHAs per house rule, so gen03-06
results remain reproducible from `main` regardless of gen07's fate.

- [ ] **B1. Counterfactual twin rewards** (PAUSED: DESIGN FORK for Kilian). Per-episode
      action-independent baseline b(t) subtracted from the per-tick latency reward. Any
      action-independent b(t) preserves the game up to a per-episode constant (verified property:
      sum_t [r(t) − b(t)] = total_wait − sum_t b(t)); the choice is purely about variance
      reduction / which uncontrollable component to strip. Feasibility confirmed: `env`
      exposes `_arrival_schedule` and per-tick `remaining_demand`, so both options below are
      clean to implement and both are numerically test-verifiable.
      - **Option A: arrival baseline** `b(t) = −cumulative_arrivals(t)`. Zero extra rollout cost
        (read from `_arrival_schedule`). Strips the arrival-driven backlog trend (the dominant
        uncontrollable term under a fixed demand seed). Does NOT remove attack damage.
      - **Option B: greedy no-attack twin** `b(t) = −remaining_demand` of a deterministic greedy
        rollout on a twin env replaying the same `_arrival_schedule` with NO attacker. One greedy
        rollout per episode (the B9.ii timing probe measures the overhead; greedy ~0.2-0.6 s/ep).
        Strips both the arrival trend AND the "unavoidable under a competent clean policy"
        component: i.e. it directly targets the M1 pathology (attack damage flooding the signal),
        because what remains is the marginal latency THIS policy incurs beyond clean-greedy.
      - **Recommendation: B** (it is the one that addresses the diagnosed SNR mechanism; A only
        removes the arrival trend, which competence already handles). Open sub-questions for the
        same conversation: (i) reference policy = greedy-insertion (recommended) vs the ε-greedy
        coping baseline; (ii) attacker reward symmetry: attacker gets the negation of the same
        difference reward (keeps zero-sum-up-to-constant) vs its own twin; recommend the former.
      Flag (either option): `--reward-baseline {none,arrivals,twin}` (default `none` = historical).
      Tests: numeric telescoping-up-to-constant on a real episode; baseline independence from the
      agent's actions; twin isolation (no state leak into the live env); default path unchanged.
- [ ] **B2. Entropy repair**: `--target-entropy-mode {lnN,absolute}` with per-decision-type
      absolute targets; separate antagonist target (the gen04b hypothesis becomes testable here);
      log per-decision-type entropy. Tests: alpha-loss sign regression, target selection.
- [ ] **B3. Exposure/strength curriculum**: episode-level attack schedule (p_attack, budget
      ramp), competence-gated ramp rule (attack strength rises only while a windowed training
      delivery/W stays inside a band). Flag: `--attack-curriculum`. Tests: schedule determinism,
      gating logic.
- [ ] **B4. Attacker learnability package + adversary population** (carries a DESIGN CHOICE for
      Kilian). Two separable parts: (i) the factored antagonist head (pick asset, then edge on
      its route; masks compose): a mechanical, additive change, low risk; (ii) the
      adversary-population training loop. **Design choice on (ii):** how rich a population?
      - **B4-lite (recommended for the thesis timeline):** the defender trains against a FIXED
        mixture of the existing scripted attackers (`targeted`, `pathrand`, `gateway`/mask-first)
        sampled per episode with logged weights. No inner BR-training loop. Cheap, deterministic,
        directly attacks the co-evolution cycling (fictitious-play flavour) without the
        compute/complexity of nested best-response training. This alone is a defensible "adversary
        population" for Obj-3.
      - **B4-full:** periodically freeze the defender, train a fresh BR attacker against it, add it
        to the population (PSRO/double-oracle). Strongest theoretically; expensive (nested training
        loops) and the gen03/04 evidence says learned BRs are weak in leashed arenas (though gen05
        says they bite under route reach, which the contested arena has). Higher risk on the
        Aug-3 calendar.
      - **Recommendation: build B4-lite now; keep B4-full as a recorded stretch** gated on the
        C1 result and remaining calendar. The factored head (i) is worth building regardless.
      Tests: mask correctness, action round-trip, population sampling reproducibility + logged
      weights.
- [ ] **B5. Credit horizon options**: γ flag surfaced (0.997+ default for gen07), optional
      n-step targets. Tests: n-step equivalence at n=1.
- [ ] **B6. Contested-resupply skin**: `--problem contested` factory (chokepoint arena reuse,
      naming, config defaults per gen07 ledger). Tests: factory smoke + config lock.
- [ ] **B7 (FUNDED, modest scope). ERB demo generator refresh**: dynamic-dispatcher demos
      (optionally under mixed attacks) for the Obj-3 ablation; reuse `generate_erb_*`
      machinery. First to drop if the calendar bites (Kilian 2026-07-06).
- [ ] **B8 (FUNDED, DEFERRED for a scoping call).** Eval-only rolling-ALNS Obj-5 reference arm.
      Deferred overnight 2026-07-06 because it carries a faithfulness judgment better made with
      Kilian: the existing `AdaptiveLargeNeighborhoodSearchVRP` is STATIC + single-depot, so a
      rolling multi-depot adaptation (re-solve the pending-request -> truck assignment at each
      decision event over current truck positions) is a substantial change to a 483-line class,
      and at capacity-1 / 2-trucks the VRP degenerates so ALNS may reduce to ~greedy sequencing (a
      weak "SOTA metaheuristic" undersells the Obj-5 comparison). Options: (a) build the faithful
      rolling-ALNS anyway; (b) raise truck capacity so the VRP is rich enough for ALNS to earn its
      keep; (c) use rolling greedy-insertion as the deterministic "reactive SOTA" reference and
      record ALNS as future work. Eval-only, blocks nothing; add before Phase C (C3). No default
      change either way.
- [ ] **B9. Pre-launch gates** (cheap, pre-registered in the gen07 ledger):
      - Suite green (≥83 tests + new ones).
      - Timing probe: s/ep for BOTH phases and the twin-rollout overhead (SYSTEM.md lesson);
        publish the compute envelope before launch.
      - Competence probe on the contested arena (greedy band, headroom, attack recoverable:
        target attacked delivery within the trainable band, not collapse).
      - **BR gate (gen04 re-run with the package)**: a retrained best-response attacker must
        beat random blocking (PASS ≥ 1.25x, as gen04). FAIL consequence pre-registered: Tier-1
        proceeds on the fitted-scripted portfolio alone; the BR failure is reported as a finding.

**Exit criteria:** all gates green + Kilian's launch approval (⛔K).

## Phase C: the gen07 campaign (MID TERM: ~Jul 19 - Aug 2)

Waves, each with a go/no-go read before the next (⛔K at each launch). Long jobs via the
detached-orchestrator pattern (gen05 recovery lesson: nohup + disown, own session). Compute
envelope finalised at B9; working assumption ~3-parallel on the M4, eval is cheap.

- [ ] **C1. Wave 1 (core):** {vanilla, sacred-curriculum} x 3 seeds; selection on validation
      attackers; per-arm BR trainings + fitted-scripted portfolio; exploitability + held-out
      portfolio eval. Interim read against the pre-registered primary. **Decision point ⛔K:**
      proceed / adjust (only via ledger amendment) / stop.
- [ ] **C2. Wave 2 (causal controls):** {dr, entropy-matched vanilla} x 3 seeds (+ ERB ablation
      arms if funded). Same pipeline.
- [ ] **C3. Evaluation-only extensions** (cheap, order by thesis value):
      budget-axis sweep curves (both registers); rolling-ALNS reference row (if B8);
      Obj-4 surrogate demo (depot grid → neural metamodel → validate argmax); ZST held-out
      geometry transfer.
- [ ] **C4. Close-out:** gen07 ledger result sections; `SACRED_PROGRESS.md` entries;
      `THESIS_STORYLINE.md` Act IV updated; **freeze (Aug 3-7 ⛔S)**; tag the freeze commit.

**Contingencies:** C1 primary null → the thesis's Act IV becomes "the fixes are insufficient;
diagnosis sharpened" (writable; pre-registered branch); timeline slip > ~4 days → drop C2
optional arms first, then C3 extensions, never the C1 core; anything threatening the Aug 7 rail
→ freeze immediately on whatever is complete (every wave is independently reportable).

## Phase D: thesis writing support (LONG TERM: Aug 8-28)

- [ ] D1. Serve the thesis planner: figures/tables strictly from ledgers; probe scripts committed
      per figure (reproducibility record); the telemetry and frontier plots.
- [ ] D2. Methods-chapter fact-checking against code (the planner's read-only questions).
- [ ] D3. Poster support (due with the thesis, 28 Aug).
- [ ] D4. Repo freeze hygiene: final `HANDOVER.md` update, env/envs merge stays POST-submission
      (TASK.md TODO), archive scratch.

## Phase E: post-thesis (LONG TERM: Sep 2026 →, all optional, all ⛔K)

- [ ] E1. Publication (parked entirely per Kilian 2026-07-06; revisit only after the thesis is
      submitted, at Kilian's initiative).
- [ ] E2. Variant B (interception/escort physics, application 3): the theoretically cleanest
      exploitability game; ~1 week build, designed in `DIRECTION.md` §3.
- [ ] E3. BLADE/Panopticon demonstration integration (industry-facing demo of the trained
      policies).
- [ ] E4. Full SBO loop (acquisition + refinement over designs) extending the Obj-4 demonstrator;
      multi-city ZST.

## Standing operations reminders (unchanged)

Single &&-chained commands for Kilian; his Mac never sleeps; he pauses runs for heat/noise;
never train without a ledger; never compare across git states; stochastic eval of max-entropy
policies; selection on validation attackers only; paste raw test output; read tfevents not logs;
time both phases before projecting; scheduled wakeups only with Kilian's permission.
