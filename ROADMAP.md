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

> **⚠️⚠️ ACTIVE PLAN = PHASE M (MULTI-CONVOY), 2026-07-07.** Single-convoy Phase I is banked (B2-P3)
> but F1 (the symmetric sweep) destabilised SACRED and single-convoy cannot meet Obj-5's metaheuristic
> clause; the active build is now MULTI-CONVOY interdiction (soft interception + a loss-averse
> mission-failure objective), oracle-proven to meet all five objectives. See
> `REDESIGN_INTERDICTION.md` §10 and Phase M below.
>
> **⚠️ ACTIVE PLAN = PHASE I (INTERDICTION) BELOW.** The redesign (`REDESIGN_INTERDICTION.md`,
> approved 2026-07-06) supersedes the contested-destination exploitability plan. Phases A-E further
> down are HISTORICAL: Phase A (sign-off, probes) and Phase B (the five learnability fixes, suite
> 109 green on branch `gen07-contested`) are DONE and partly carry over; Phase C (the gen07
> destination matrix) is SUPERSEDED by the flat-landscape finding and never ran. The new build is
> Phase I.

## Phase I: the interdiction-game build (ACTIVE, 2026-07-06 → Aug 3 freeze)

Goal: the positive thesis result, convoy routing as a Stackelberg interdiction game where SACRED
learns a mixed-strategy route policy that is less exploitable to interception than shortest-path
and vanilla SAC, approaching the computable minimax equilibrium. Kaliningrad graph, single convoy
first (Kilian 2026-07-06). Full design: `REDESIGN_INTERDICTION.md`. Pre-registration:
`experiments/gen08_interdiction.md`. New code on a branch off `main` (e.g. `gen08-interdiction`);
the gen07 fixes on `gen07-contested` are cherry-picked in as needed (twin/counterfactual reward,
evaluation discipline). Separation policy from the redirection still holds (`main` frozen for
`src/`; regression-guarded flags; historical modes reproduce).

- [x] **I0. Equilibrium oracle: DONE 2026-07-06** (`src/baselines/interdiction_oracle.py`, +7 tests,
      commit after the redesign docs). build_interdiction_game + solve() → minimax value (loss_mixed),
      loss_det, both equilibrium mixed strategies; best_response_attacker + interception_of_distribution
      (exploitability of any learned defender); route_distribution_from_first_hops (next-hop policy →
      route mixture). The ground truth SACRED is validated against.
- [x] **I1. Interdiction env CORE: DONE 2026-07-06** (`src/envs/interdiction.py`, +4 tests). Attacker
      commits K interdiction assets (hidden); defender picks a route via first-hop node (reuses
      node-selection); interception terminal + travel-cost reward (zero-sum). **G1 env-fidelity gate
      PASSES**: the env reproduces the oracle's loss_det and loss_mixed end-to-end (Monte Carlo) on
      synthetic AND Kaliningrad 33→71 (gap ≥ 0.8). Suite 120 green.
- [x] **I1b. SAC-trainable env: DONE 2026-07-06** (`make_interdiction_env` + masks, +2 tests).
      GraphEnv-backed `observe()` yields the observation the existing `featurize_state` consumes;
      defender mask = first-hop nodes (reuses node-selection), attacker mask = candidate edges
      (reuses the antagonist's edge-selection). Verified ProtagonistSAC/AntagonistSAC act on it and a
      sortie resolves. Suite 122 green. (Multi-branch next-hop physics deferred; first-hop = route
      for the disjoint-route single-convoy headline.)
- [x] **I2. Feasibility slice: DONE + PASSED 2026-07-06** (`scripts/train_interdiction.py`,
      `experiments/gen08_interdiction.md` G2). Defender SAC vs the ORACLE best-response interdictor
      (fictitious play on the empirical average). **First positive result: shortest_path 1.000 >
      vanilla 0.275 > SACRED 0.235 (equilibrium 0.167); adversarial training cut interception
      100%->23%.** Paid-for gotchas: reward_scale 0.001->1.0 (default swamps the signal);
      best-respond to the empirical AVERAGE play, not the instantaneous policy (else it oscillates).
      Caveat driving I3: the instance is SYMMETRIC (uniform equilibrium) -> thin SACRED-vs-vanilla
      gap; the clean contrast here is vs deterministic shortest-path.

- [ ] **I3. The experiment matrix** (⛔K launches; the thesis's positive Results act). Build on I2.
      **I3a BUILD DONE 2026-07-06** (decisions Kilian 2026-07-06: class (c) first, length-derived
      band): heterogeneous edge vulnerability = soft interception end-to-end
      (`length_band_vulnerability` + `survival_intercept_fn` in the oracle; `edge_vulnerability` +
      seeded Bernoulli `resolve()` in the env; `--edge-vuln-band` + three-way exploitability
      reporting (policy | trailing-window | all-history) + `--json-out` in
      `scripts/train_interdiction.py`). Closed form verified (disjoint routes: d_i ~ 1/p_i*, value
      1/sum(1/p_i*)); G3 fidelity gate PASSED; suite 127 green. Instances pinned by
      `scratch/vuln_band_probe.py`: PRIMARY = 33->71 band (0.15,0.95) K=1 (equilibrium 0.063,
      uniform 2.51x suboptimal, shortest 0.449). Pre-registration drafted in the gen08 ledger
      (I3 section); launch waits on Kilian.
      1. **Asymmetric instances (the priority): open the SACRED-vs-vanilla gap.** The symmetric
         6-disjoint-route game makes vanilla's max-entropy mixing incidentally near-optimal. Create
         NON-UNIFORM equilibria so vanilla (uniform-ish) is measurably exploitable and SACRED must
         learn the specific mixed strategy: (a) ~~K>=2 interdictors~~ **CORRECTED 2026-07-06: K>=2
         alone is NOT an asymmetry source (on disjoint routes with hard interception the equilibrium
         is uniquely uniform for every K: best response = the top-K defender masses); K stays the
         budget/sweep axis**; (b) shared-edge / non-disjoint candidate routes: **BUILT 2026-07-06
         (B2): route-walk trie (`begin_walk`/`step_walk`/`walk_distribution`, exact branch-product
         policy mixtures), `--route-mode walk`, TAP metric (trailing-averaged policy distribution),
         `cost_constrained_value` frontier LP; suite 131 green; B2 pre-registered in the gen08
         ledger (33->71 k8 primary: equilibrium 0.167 vs best-possible cost mixture >= 0.467:
         the wave-1 imitation escape is closed by construction); LAUNCH AWAITS KILIAN**;
         (c) heterogeneous route vulnerability: **BUILT + RUN (I3a wave 1: primary FAILED, headline
         replicated, mechanisms ledgered; class (c) exhausted on disjoint routes by the
         descending-band probe)**. The oracle handles all of these (`build_interdiction_game` with
         any K, any route set, any `intercept_fn`).
      2. **Arms x seeds:** shortest_path, vanilla, sacred, + the equilibrium oracle (ground truth);
         >=3 seeds; report exploitability (best-response) + distance-to-equilibrium, dual-level stats.
      3. **Sweeps as curves:** interdiction budget K, edge-connectivity of the OD pair.
      4. **Learned-antagonist co-evolution (Obj 1):** replace the oracle interdictor with an
         antagonist SAC (edge-selection; the env already exposes the antagonist mask). Reference the
         SMDP trainer's antagonist transition enrichment (`src/agents/sacred_atla.py` ~L197-220) for
         the transition format. Show co-evolution -> equilibrium alongside the oracle result. This
         is the full ATLA demonstration; the oracle version is the strong-adversary baseline.
      5. **Refinements from I2:** trailing-window average (closer to loss_mixed than all-history);
         confirm vanilla collapses to deterministic on asymmetric instances (higher exploitability).
      Ledger: extend `experiments/gen08_interdiction.md` (or gen09 for the matrix); pre-register the
      metric/gates before launching, per house rule.
- [ ] **I4. Objective extensions** (eval-mostly): Obj-4 interdiction-aware base/FOB placement
      (site for egress connectivity; surrogate over a placement grid, validate argmax); ZST (transfer
      the mixed policy to a held-out OD pair / theatre); Obj-3 ERB from shortest-path/equilibrium.
- [ ] **I5. Multi-convoy richness (LATER, after the single-convoy matrix lands):** multiple
      convoys/FOBs, coordinated routing spreading interdiction risk (the VRP flavour).

**Status (2026-07-07, end of the B2 arc): THE HEADLINE IS BANKED.** I3.1 concluded with the
programme's first pre-registered sacred-vs-vanilla PASS (B2-P3, smooth fictitious play, SHA
`874d3f3`): TAP ladder **shortest_path 1.000 > vanilla 0.477 > uniform 0.455 > SACRED 0.362 >>
equilibrium 0.167** (3/3 seeds + pooled, every clause). Dynamics work CLOSED by Kilian's
pre-committed exit criterion.

### Findings to date (the citable spine; numbers only from `experiments/gen08_interdiction.md`)

1. **Equilibrium level (proved before training):** deterministic routing is fully exploitable
   (loss_det 1.0); the minimax mixed strategy cuts interception to 0.167-0.33 on the real graph.
2. **I2 (symmetric slice):** adversarial RL reaches 0.235 vs shortest-path 1.000 (100% -> 23%).
3. **Instance structure decides the control's fate:** on cost-correlated vulnerability instances
   vanilla IMITATES the equilibrium (wave 1 primary failed); on shared-edge instances no
   cost-driven mixture can (oracle bound 0.467 vs equilibrium 0.167) and vanilla lands ABOVE
   uniform noise: cost-calibrated mixing is predictability with extra steps.
4. **The FP dynamics bracket (measured):** latest-pure-BR over-disciplines (last-iterate
   cycling, good average 0.24-0.26); all-history mixture under-disciplines (stale -> cost
   gradient parks the policy, entropy 2.0 -> 0.7); SMOOTH fictitious play (softmax BR to
   trailing-250 play, tau 0.05 probe-pinned) is the stable middle and PASSED.
5. **Estimator lessons:** trailing-window/mid-cycle readings are biased against FP learners; TAP
   (trailing-averaged policy distribution) is the deployable estimator; smokes validate
   plumbing, not slow dynamics (1000-sortie drift signature required).
6. **Honest open gap:** strong form unmet: SACRED lands ~half-way between uniform and the
   equilibrium (distance 0.163-0.239) on the policy form; the average-play reading (0.26-0.28)
   is closer. Closing it is future work (annealed smoothing / optimistic dynamics), NOT a
   pre-freeze task.

### Future work (agreed shape, 2026-07-07; every launch ⛔K)

**SHORT TERM (now -> Aug 3 freeze; experiments, in priority order):**
- [ ] **F1. Waves A + C (pre-registered, unlaunched):** hard-instance K sweep {1,2,3} x 3 seeds
      + 110->135 connectivity contrast (~5.5 h serial): completes Obj-5's "varied disruption"
      curves and replicates the headline across K. Ready to launch as ledgered.
- [ ] **F2. Learned-antagonist co-evolution demonstration (I3.4, Obj 1/3):** antagonist SAC
      (edge-selection; env mask exposed; transition enrichment per `sacred_atla.py` ~L229-252)
      replacing the oracle interdictor on B2-P; evaluate with the ORACLE BR regardless
      (portfolio-max). One instance x 3 seeds as a demonstration, not a matrix.
- [ ] **F3. Obj-4 demonstrator (oracle-driven, eval-only):** interdiction-aware base/FOB
      placement: equilibrium interception per candidate OD/placement over a grid + small
      surrogate + argmax validation. An afternoon; zero training.
- [ ] **F4. ZST (aim-level):** transfer the trained B2-P3 policy to a held-out OD pair; score
      vs that pair's oracle. Eval-only afternoon.
- [ ] **F5 (drop first if the calendar bites): ERB ablation** (seed from shortest-path or the
      equilibrium mixture; time-to-competence, modest scope).

**MID TERM (Aug 3 -> Aug 28: writing wins every conflict; Phase D below):** figures/tables
strictly from the gen08 ledger (the ladder, the cost-security frontier via
`cost_constrained_value`, the FP-bracket trajectories, the equilibrium maps); methods
fact-checking; poster; final HANDOVER refresh + freeze tag.

**LONG TERM (post-submission; Phase E below):** close the strong-form gap (annealed
smoothing/optimistic or extragradient dynamics -> last-iterate to Nash); multi-convoy richness
(I5: coordinated sortie spreading); K>=2 co-evolution; publication (Kilian's call, parked);
BLADE/Panopticon demo; full SBO loop.

## Phase M: multi-convoy interdiction (ACTIVE, 2026-07-07 -> Aug 3 freeze)

**Why (supersedes the single-convoy F1 sweeps):** single-convoy cannot meet Obj-5's metaheuristic
clause (ALNS degenerates to shortest-path) and the symmetric single-convoy sweep destabilises SACRED
(F1 killed). Under Kilian's "make SACRED work" mandate the programme extends to MULTI-CONVOY
interdiction. Design: `REDESIGN_INTERDICTION.md` §10. Oracle findings + forward record:
`experiments/gen08_interdiction.md` (multi-convoy pivot section). Single-convoy B2-P3 stays the
banked headline; this is the extension that meets all five objectives and wins bigger.

- [x] **M0. Oracle proof: DONE 2026-07-07** (`scratch/multiconvoy_{probe,scan,spectrum,cost}.py`, NO
      training). Multi-convoy + SOFT interception + a LOSS-AVERSE (mission-failure) objective: SACRED
      gap median 0.48 (N=2) across 20 OD pairs, growing with fleet size; the deterministic coordinator
      (a real ALNS problem) is beaten on its cost-security frontier; a risk-neutral objective is the
      trap; boundary K < #routes. ALL FIVE objectives confirmed meetable.
- [ ] **M1. Multi-convoy env** (⛔K build): N convoys, joint first-hop/walk routing, hidden K-asset
      interdictor, SOFT interception (per-edge survival), mission-failure reward (loss if ANY convoy
      lost). M0-style fidelity gate vs the multi-convoy oracle. Reuses the single-convoy env + oracle.
- [ ] **M2. ALNS fleet-coordination baseline** (Obj-5 metaheuristic): coordinate N convoys' routes
      minimising cost-vs-risk = the non-degenerate classical opponent (the oracle's loss_det); plus a
      shortest-path/greedy reference.
- [ ] **M3. Train SACRED vs the interdictor** (⛔K launch; pre-register a gen09 ledger): confirm
      SACRED LEARNS the ~0.31 mission-failure mixed strategy vs ALNS's ~0.8 and vs a non-adversarial
      SAC, on a probe-selected high-headroom instance; seeds; best-checkpoint; an entropy floor to
      avoid the symmetric-instance collapse.
- [ ] **M4. Sweeps + objectives:** N / K / connectivity curves (varied disruption, Obj-5);
      learned-antagonist co-evolution (Obj-1/3); Obj-4 placement + fleet size; ZST. Each launch ⛔K.

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

**Separation policy (Kilian 2026-07-06; UPDATED at gen07 close).** Original: `main` stays the
frozen campaign record during the gen07 exploration; all gen07 code on branch `gen07-contested`,
flag-gated with defaults preserving historical modes + regression tests; ledgers pin SHAs so
gen03-06 stay reproducible. **Update (gen07 closed → redesign):** gen07's exploration concluded
(the flat-landscape finding), so the frozen-main caution has served its purpose; `gen07-contested`
(tested 109 green, additive, flag-gated) was **fast-forwarded into `main`** so there is a single
authoritative branch for handover, gen03-06 reproducibility is intact (SHAs pinned; historical
modes preserved by flags + tests). The new interdiction build (Phase I) branches from `main` as
`gen08-interdiction`; the same discipline applies (flag-gated, tested, ledger-pinned SHAs).

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
