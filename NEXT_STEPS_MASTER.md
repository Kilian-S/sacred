# NEXT_STEPS_MASTER.md: the agreed work programme after 2026-07-12 (checklist for incoming agents)

> **⚠️ UPDATE 2026-07-16: BLOCK R (below, before §0) IS NOW THE ACTIVE FRONT.** The
> disjoint-route baseline finding (`CRITIQUE_16-07-26.md`, verified oracle-exact by
> `scratch/disjoint_baseline_probe.py`) showed the static K=1 ladders' comparative claims are
> refutable by a two-line max-flow heuristic; Kilian's decision 2026-07-16 is to REPAIR the
> record and RELOCATE the positive claims via the Block R rescue programme. **The overarching
> goal (Kilian, verbatim standing mandate): SACRED must end with a positive, scientifically
> valid and interesting claim.** Blocks A/B below remain complete-and-banked history; Block C
> (FAR/writing) rails still bind.
>
> **Provenance.** Agreed between Kilian and the outgoing Fable instance on 2026-07-12 (Fable's
> last session). This file merges the tiered programmes of `CRITIQUE_EXAMINER.md` §6 and
> `CRITIQUE_12-07-26.md` §6-7, with Kilian's explicit ordering decision: **computational testing
> comes FIRST; the Final Activities Report, the interactive exhibit and the THESIS_STORYLINE
> rewrite move to the BACK.** Exact dates do not matter; the ORDERING does, except for two hard
> external calendar rails that bind regardless of position (see "Calendar rails").
>
> **This is the active plan.** Work through it as a checklist, top to bottom within each block.
> Items tagged [eval-only] or [oracle-only] need no training and may run CONCURRENTLY with a
> training item; the working agent decides what to parallelise. Tick items off here (append a
> one-line result pointer per item) as they complete.
>
> **PROGRESS (2026-07-12, autonomous session under Kilian's standing launch authority):**
> - [x] **A0** DONE: rows folded into gen13/gen14 ledgers.
> - [x] **A1** DONE (`experiments/gen24_distill.md`): select-on-train primary fires for
>   adversarial (distill overfits to 2.07 without an external signal), but the symmetric
>   val-early-stop row REVERSES it (distill 1.555 < adversarial 1.761). ZST act re-scoped:
>   adversarial training's value = label-free + self-stopping, NOT superior transfer.
> - [x] **A2+A3** DONE (`experiments/zst_map_robustness.md`): shuffled-reality bar PASSED
>   (1.80x vs random 2.19x); intel-error curve flat (full obs shuffle +0.03); constant-map
>   diagnostic shows per-edge map-reading is NOT the mechanism (info-free map +0.09). Wording
>   rule recorded ("geometry-informed, threat-robust hedge").
> - [x] **A4** DONE (`experiments/gen25_dr_control.md`): vanilla n=3 = 2.354 +/- 0.014 (tight;
>   worse than random-init); DR control 2.056 ~ random-init => BEST-RESPONSE pressure is the
>   causal ingredient (pre-registered branch fired). Note: first launch was reaped (gen05 lesson);
>   relaunched detached. **BLOCK A COMPLETE.**
> - [x] **A5** DONE (`experiments/d3_gdansk.md` appendix): the 0.109 was seed-0-specific
>   (seeds 1/2: 0.443/0.433); cross-seed reliability 0.32-0.54; poster claim DOWNGRADED.
> - [x] **A6** DONE (`experiments/a6_a7_a8_completions.md`): retrieval MATCHES the generalist
>   (1.676 vs 1.677/1.733); transfer ladder fully bounded.
> - [x] **A7** DONE (same ledger + `assets/transfer_gap_closure.png`): gap-closure ladder
>   0.90 -> 0.54 -> 0.45 -> 0.20 (Istanbul) -> 0.04 (Kyiv); far-end wording rule recorded.
> - [x] **A8** DONE (same ledger + `assets/prevalence.png`): 69% of 160 ODs have det/eq >= 2;
>   headlines sit in the top decile BY SCREEN DESIGN.
> - [x] **B1** DONE (gap 0% actor0 / +19.3% actor1; joint = safe default; strict bar not met) (`experiments/b1_integration_gap.md`): joint-vs-sequential design gap,
>   two actors, in flight.
> - [!] **B2** HARNESS READY (`experiments/b2_llm_benchmark.md`, dry-run validated). **UNBLOCKED
>   2026-07-16: a LOCAL LLM workbench exists** (Prof. Angeloudis's box `cv-iits-w05`, no API keys /
>   no spend). SSH `killian@100.88.32.88` (double-L username), gateway on :8080 reached via SSH
>   tunnel `ssh -N -L 18080:localhost:8080 killian@100.88.32.88` -> `http://localhost:18080/v1`, key
>   `iits-local-key`; live model `llama-3.3-70b` (+ `qwen3-27b` one command away). Full details in
>   the HANDOVER top banner. TODO before the live run: (a) Kilian's go (it hits the shared GPU box);
>   (b) add a generic OpenAI-compatible base-url path to `scratch/b2_llm_benchmark.py` (the openai
>   branch already posts to /v1/chat/completions, just parameterise the host).
> - [x] **B3** DONE (`experiments/b3_b4_oracle.md`): three-regime risk-aversion law (mission =
>   the unique objective determinism cannot escape by spreading).
> - [x] **B4** DONE (same ledger): multi-OD correlation gap median 14.4% (bar met; Tier-3
>   multi-OD game justified).
> - Block C: NOT STARTED (paused per Kilian's instruction; the FAR's 30 July external deadline
>   stands regardless).

---

## Block R (2026-07-16, ACTIVE): the claims-repair + rescue programme

> **Provenance.** Planned by the critic instance at Kilian's request 2026-07-16, after the
> disjoint-baseline finding (`CRITIQUE_16-07-26.md` §1; probe `scratch/disjoint_baseline_probe.py`,
> commit `e00cb37`). Design logic: the two-line max-flow heuristic (uniform/inverse-vuln stack over
> the edge-disjoint routes) matches or beats every trained static K=1 number, so every rescue
> experiment targets one of its three provable blind spots: (R1) BUDGET SATURATION (K approaching
> the min-cut m, where the heuristic degrades AND exact solvers become infeasible), (R2)
> ADAPTATION (a pattern-of-life adversary, against which every static mixture is capped at
> iid_eq), (R3) COORDINATION (multi-OD corridor sharing, where the heuristic is an independent
> product and provably leaves the B4-measured 14.4% on the table). House rules unchanged and
> HARD: each item opens its own pre-registered `experiments/<gen>.md` (question, bars, pinned
> SHA) BEFORE any CPU; oracle/eval probes are free; NO training launch without Kilian's explicit
> in-conversation go; suite green after any `src/`/`scripts/` change; failures reported plainly
> (every fail branch below is a writable, measured boundary).
>
> **The target claim if R1 and R2 land (the thesis's new positive spine):** *below a measurable
> boundary, contested routing needs no learning (we prove it: the two-line heuristic is
> near-optimal); at the interdiction budget's saturation point, against adaptive adversaries,
> and under joint coordination, both the heuristic and the exact solvers fail, and adversarial
> self-play is the only method that delivers, demonstrated against computable yardsticks.*

> **PROGRESS (2026-07-16, autonomous session, Kilian's full launch authority):**
> - [x] **R0a/R0b/R0c** DONE (commit `0251dda`): rows + fleet-cost columns folded into all seven
>   ledgers with the binding wording rule; structure-discovery row (policy disjoint-core mass
>   0.62 vs eq 0.703 vs uniform 0.333; zero-shot tracks per-instance eq); boundary screen
>   (35-159 saturates at K>=4; shortlist -> 71-33 m=6; population 5% of ODs >= 1.5x at K=1).
> - [x] **gen26 step 1** PASSED + STRONG (`c9c474a`, ledger): K=3 n=3 = 0.664 +/- 0.018 < both
>   heuristic variants (0.738 uniform / 0.737 inv-vuln); eq 0.604.
> - [x] **gen26 step 2** DONE (`77fe57f`): greedy-BR mode, suite 167, fidelity <= 1.8% at K<=3.
> - [x] **gen26 COMPLETE** (`505c466` + second-pass amendments in the ledger): K=5 PRIMARY PASS
>   (0.667 +/- 0.016, 3/3 < uniform-disjoint 0.705; STRONG < inv-vuln 0.638 not met); K=6 single
>   seed 0.718 beats both variants; boundary map `assets/k_boundary_map.png`. TEMPERED by the
>   second pass: full-menu uniform TIES at K=5 (0.666); tabular-FP+greedy-BR BEATS (0.621/0.690);
>   surviving claim = the boundary map. OPEN: K=6 to n=3 + full-menu rows before thesis sentences.
> - [x] **gen27 seeds PASSED, PRIMARY + STRONG 3/3** (ledger): pooled held-out 0.639 +/- 0.025 vs
>   the static cap; beats the MEASURED local static optimum everywhere; full-menu anti-repeat
>   fails (1.37x); composed disjoint+anti-repeat (0.50-0.61) bounds below (binding wording in the
>   ledger); worst-case premium 1.57x (regime-conditional scope sentence). OPEN: the no-window
>   causal control (training overnight) + its ledger fold.
> - [~] **B2 LIVE** in the original conversation: design finalised (llama+qwen, unhinted, 3
>   instances, on-box tmux runner); first llama transcript reviewed-able at
>   `scratch/b2_livetest_llama_transcript.txt`; gateway direct at http://100.88.32.88:8080/v1.
> - [>] **R3-air HANDED OFF** to a fresh instance: `AERIAL_BRANCH_HANDOFF.md` (trained aerial =
>   MUST-HAVE, Kilian 2026-07-16).
> - [!] **SECOND CRITIC PASS (2026-07-16, `scratch/critique_followup_probes.py`; ledger
>   amendments in gen26/gen27, the gen27 one landed BEFORE its results were read):** (i) gen27
>   gains the naive-DYNAMIC row — a two-line anti-repeat heuristic beats iid_eq by ~2x
>   (0.50-0.61x) on all 6 held-out ODs, so the act's wording must clear that row, not iid_eq;
>   (ii) gen26's K=5/6 cells gain FULL-MENU heuristic rows (uniform-full 0.666 TIES SACRED at
>   K=5; at K=6 the best naive is 0.730 vs SACRED 0.718, 1 seed) and a TABULAR-FP-with-greedy-BR
>   row that BEATS SACRED at both cells (0.621/0.690) => the "only self-play can train there"
>   wording is retired; the K=6 "sole survivor" point needs n=3 + the full-menu rows before any
>   thesis sentence. Rescued-claim centre of gravity: boundary map + label-free amortisation
>   (gen27), not single-instance superiority.

> **⚠️ BLOCK R CLOSE-OUT (2026-07-19, the critic instance; this block is now HISTORY).**
> B2 COMPLETE (two instances, both models; `experiments/regime_decision_table.md` = the
> synthesis). R3-air CLOSED (gen28: three measured negatives + the screen/exhibit products;
> the fleet Tier-1 wording RETIRED by the 2026-07-19 baseline-completeness appendix in the
> aerial ledger). The CLOSING EXPERIMENT gen29 (three-stream coordination, worktree
> `../sacred-gen29`) RAN AND FAILED both tiers with a clean blinded control, while its oracle
> screen banked the project's only complete-baseline-proof gap (median 31%). OPEN: the gen29
> re-aim/distillation/close decision (Kilian's). **The active front is now Block C — the FAR
> (hard 30 July) and the boundary-map re-spine — plus, optionally, the oracle-only
> security-aware facility-location act (supervisor direction 2026-07-19).** See the HANDOVER
> 2026-07-19 banner for the three-worktree map; the items below are the historical record.

### R0. Repair + aim (oracle/eval-only, FREE, no go needed; ~1 day; do first, in this order)

- [ ] **R0a. Fold the disjoint rows into every ladder** [eval/oracle, ~half day].
  Extend `scratch/disjoint_baseline_probe.py` to emit, for BOTH headline instances, the gen12
  sweep grid, the gen16/gen22 held-out pools and the A8 population: (i) uniform-disjoint-stack,
  (ii) inverse-vulnerability-disjoint-stack, (iii) each variant's expected fleet cost (the
  fairness column: disjoint routes include long detours; SACRED's 123.1 vs eq 120.8 vs ALNS 96.1
  are the anchors). Fold as dated appendices into gen13/gen14/gen12/gen16/gen22/a6_a7_a8
  ledgers + amend the B2 anchor table (register (b) must carry the heuristic row BEFORE any live
  LLM call). Add the binding wording rule to each: no "beats every uncalibrated strategy class" /
  "standard algorithms cannot achieve" sentence survives unqualified. GAIN: the ambush becomes a
  disclosed row; all later experiments score against the right baseline.
- [ ] **R0b. The structure-discovery row** [eval-only, ~2-3 h]. Load the banked best-checkpoint
  actors (gen14 MC n=10, gen14 SC n=10, gen16 generalist n=3) and measure the policy mass placed
  on the disjoint core vs the padded duplicates, per act (descriptive, no bar). GAIN: the
  measured basis for the surviving positive sentence: *self-play discovers the independent-route
  structure without being told it, without labels and without a solver* (the heuristic cannot
  cheapen this: it is TOLD the structure).
- [ ] **R0c. The boundary screen (the aiming step)** [oracle-only, ~half day]. Over the A8
  population (160 ODs, 4 cities) compute heuristic/eq as a function of K in {1, 2, 3} (exact)
  and, at K in {4, 5}, the heuristic's and det's values under the verified greedy BR (exact eq
  does not exist there: that is the point). Deliverables: (i) the boundary-map figure
  (heuristic suboptimality vs K/m: thesis material in itself); (ii) the gen26 step-3 instance
  shortlist, screened by the NEW criterion (heuristic-gap at high K, not det/eq: the old
  screens measured where determinism fails, not where naive randomisation fails); prefer
  m = 5-6 instances (e.g. 33-71-class) so K = 4-5 sits at-but-not-past saturation. (iii) ~40
  multi-OD triples for R3's shortlist (B4 machinery). GAIN: the new headline is staked where
  the gap provably exists BEFORE any training.

### R1. gen26: the K-to-min-cut act [TRAINING, ⛔K per step; the static-headline rescue]

- [ ] **Step 1 (fund first; ~15-20 min wall at 3-parallel + ledger).** n=3 the K=3 cell on
  35-159 (the gen12 cell config VERBATIM: k8 menu, band 0.15-0.95, N=3, fleet-route, smooth FP
  tau 0.05, 1200 sorties, exact estimator, per-eval ckpts).
  > **PRE-REGISTERED BAR:** mean exact best-checkpoint TAP < the heuristic's 0.738 on >= 2/3
  > seeds AND pooled (current single-seed evidence: 0.661). Report beside eq 0.604 and det
  > 0.933. PASS = the crossover is real at n=3; FAIL = the crossover was seed noise, R1 step 3
  > is re-aimed by R0c or dropped (a measured boundary either way).
- [ ] **Step 2 (build, ~1-1.5 days incl. tests; no CPU risk).** Wire the VERIFIED
  `greedy_br_attacker` (A4-core, exact at K <= 2, matrix-free to K = 5 in ~8 s) into
  `train_multiconvoy.py`'s attacker refresh + the exploitability eval; gate the eager
  `obj_matrix` build behind K <= 3; regression tests: (i) greedy-vs-exact agreement in-trainer
  at K <= 2, (ii) byte-identical behaviour with the flag off; suite green, raw output pasted.
  This is A4's deliberately-deferred step, now justified by necessity (it is the ONLY route to
  any K >= 4 cell).
- [ ] **Step 3 (the headline cells; ~1 evening at 3-parallel).** Train K=4 and K=5 on 35-159
  AND one R0c-screened second OD (3 seeds on the headline cell, 1 elsewhere). Yardstick: ALL
  arms (det/ALNS plan, heuristic variants, SACRED) scored under the SAME greedy BR; the
  certified interval [v_greedy, v_greedy/(1-1/e)] reported for absolute statements.
  > **PRE-REGISTERED BAR (primary):** SACRED best-ckpt < uniform-disjoint-stack under the
  > common greedy yardstick, >= 2/3 seeds + pooled, on the headline cell. **STRONG:** also
  > < the inverse-vuln variant. **FAIL branch (writable):** "past saturation the learner no
  > longer beats naive disjointness" = the boundary map's upper edge, measured.
  GAIN if it lands: the rescued static headline: **"trained where neither exact solvers (RAM
  wall, no labels) nor naive heuristics (coverage saturation) can follow"**: a claim unique to
  self-play by construction.

### R2. gen27: the dynamic generalist (gen19 x gen16) [TRAINING, ⛔K; the ZST rescue]

- [ ] **Build (~1 day).** Extend `train_generalist.py` with the gen19 mechanism per instance:
  adversary = analytic softmax-BR (w=3, tau=0.15, the gen19 operating point, sensitivity grid
  already banked) to each instance's OWN realised-route window; observation = per-route recent
  window frequency as the third route-feature column (head-term lr 3e-2), riding per-transition
  as the menus already do. Episodes = S=40 sortie chains, gamma 0.95 (gen19 values). Oracle
  yardsticks per pool/test instance at build time: iid_eq + history_opt by RVI (the
  `within_episode_screen` machinery; the gen19 numbers reproduce as the sanity row). Pools =
  the gen16 recipe VERBATIM (kaliningrad + east_london + istanbul train, GDANSK held out,
  pool-seed 0). Timing probe + 240-sortie smoke gate BEFORE the batch (the gen19 smoke passed
  its primary at 240 sorties; expect the same signature: window-feature weight strongly
  negative = anti-repeat).
- [ ] **Run (3 seeds + 1 no-window control seed; ~half-1 day of machine).** Budget ~12,000
  sorties/seed (the gen16 budget; refine from the timing probe), select-on-train, per-eval ckpts.
  > **PRE-REGISTERED BARS. PRIMARY (the unique claim):** held-out-GDANSK stationary-tail
  > per-sortie mission failure < that OD's iid_eq on >= 4/6 ODs AND pooled, on >= 2/3 seeds.
  > Beating iid_eq means beating EVERY static object: the disjoint heuristic, the LP mixture
  > and any distilled/retrieved policy are ALL capped at iid_eq by construction: no wording
  > escape needed. **STRONG:** pooled within 2x of history_opt. **CAUSAL CONTROL:** the
  > no-window arm must land ~iid_eq (the gen19 control landed 0.148 vs 0.147). **REPORTED ROW
  > (not gated):** worst-case = each OD's marginal route mixture under its oracle BR vs V_eq
  > (gen19's was 1.06x; zero-shot will be looser; report honestly).
  > **Branches:** PASS = the crown jewel: *zero-shot transferable DYNAMIC hedging: one policy
  > outfoxes an adaptive adversary in a never-seen city, provably beyond any static method*
  > (the aim's ZST sentence becomes literally true). PARTIAL (train cities yes, held-out no) =
  > dynamic hedging learned, transfer boundary measured. FAIL = gen19 stays the single-instance
  > positive; the boundary is the result.
- **Deliberately out of scope** (recorded): the K/N-shift rows and Istanbul rotation for the
  dynamic policy (post-freeze register), full B1 Poisson demand.

### R3. gen28: the multi-OD game [TRAINING, ⛔K; gated on R0c + calendar; ~3-4 days]

Two-destination corridor-sharing game (N=2, one convoy per destination, K=1 over the union):
env = sequential per-convoy routing with the earlier route observed (the existing multiconvoy
pattern); oracle = B4's joint LP over R1 x R2 (exists). Ladder per instance: det pair / best
INDEPENDENT product (the heuristic's class, upper-bounded by B4's local search, disclosed) /
SACRED / joint equilibrium. Primary: SACRED < best independent product, >= 2/3 seeds, on an
R0c-screened triple (Jaccard >= 0.15, where B4's gap concentrates). Known risk: the gen18
exploration boundary; mitigation: coordination lives inside ONE policy's sequential joint
action (not independent followers); gate on a smoke. GAIN: a second heuristic-proof claim +
material movement toward the VRP of the title. **Drop first if the calendar bites.**

### R4. Riders (cheap, in the gaps)

- [ ] **R4a. Cost-security frontier row** [eval-only, hours]: the heuristic's fleet cost beside
  SACRED's 123.1 / eq 120.8 / ALNS 96.1 on both headlines (if the heuristic pays a material
  detour premium, "equal security at lower cost" is a surviving K=1 sentence; either way honest).
- [ ] **R4b. B2-live on the workbench** [eval-only, ~1 day; ⛔K: shared GPU box]: with R0a's
  amended anchors + the new scored question ("does the model discover independent-route
  reasoning?"); the two standing TODOs (generic base-url path; Kilian's go) unchanged.
- [ ] **R4c. The C4 re-spine** absorbs Block R's outcomes (Act I negative campaign; Act II
  dynamics account + bug arc; Act III the dynamic register, gen19 + gen27; Act IV the boundary
  map with gen26; the disjoint concession stated FIRST, on our terms).

### Block R ordering, calendar and drop order

R0a-c today/tomorrow (free) -> gen26 step 1 (first funded run; result within ~1 h of go) ->
gen26 step 2 build while step-1 verdict settles -> gen27 build in parallel with gen26 step-3
runs -> gen27 run -> R3 only if both land early -> R4b in the gaps. **Autonomous elapsed
estimates (single M4, timing probes refine after smokes): gen26 complete ~2-2.5 days from go;
gen27 complete ~2-3 days from ITS go; overlapped total ~4 days => both banked ~20-21 July**,
leaving the FAR (hard 30 July) and freeze (3 Aug) rails comfortable. Drop order if the
calendar bites: R3, then R4b, then gen26 step 3's second OD (keep the 35-159 cells), then the
STRONG bars. **Never drop:** R0a-c, gen26 step 1, gen27's primary, R4c. Those decide whether
the thesis's positive claim exists and survives review.

---

## 0. How to use this file (new agent: start here)

**Read order for onboarding:** `HANDOVER.md` top banner (project state) -> `CRITIQUE_12-07-26.md`
-> `CRITIQUE_EXAMINER.md` (the two 2026-07-12 critiques this plan implements) ->
`NEXT_STEPS_11-07-26.md` (the prior list; mostly done) -> the ledgers under `experiments/` as
needed. Numbers live ONLY in the ledgers; prose documents carry pointers.

**House rules (unchanged, HARD):**
1. NEVER launch a training run without Kilian's explicit in-conversation go. Oracle and
   evaluation probes are free.
2. Every item below opens its own pre-registered ledger (`experiments/<gen>.md`: question, metric,
   pass/fail bars, pinned SHA) BEFORE any CPU is spent. Results are appended, never rewritten.
3. Suite green (`PYTHONPATH=. pytest tests/`, currently 161) after touching `src/` or `scripts/`;
   paste raw output.
4. Prose + firm recommendation to Kilian; no multiple-choice prompts. Plan first, never dive in.
5. Report failures plainly. A failed pre-registered bar is a citable result, not a problem.

**Calendar rails (fixed, external, bind regardless of ordering):**
- **Final Activities Report + presentation: due 30 July 2026** (item C2 must be complete before
  this date even though it sits at the back of the ordering).
- **Experimental freeze: 3 August 2026, HARD.** After it, writing wins every conflict.
- **Thesis (12,000 words) + poster: due 10:00, 28 August 2026.**

**Why this plan exists (one paragraph of context):** the project has two banked headline ladders
on corrected code, a zero-shot transfer (ZST) arc across cities, a full SBO stack, and
mechanism-backed negatives. The remaining risk is not the evidence but WHAT THE CLAIMS ARE ALLOWED
TO SAY. Block A closes the specific attacks an examiner can still make (each item defends a named
claim); Block B adds the two differentiators Kilian wants (LLM benchmark, holistic SBO) plus two
cheap immunisers; Block C is the deliverables and writing work, deliberately moved last.

---

## Block A: claims-defence (computational testing; do these first)

### A0. Fold the already-measured probe rows into the ledgers [doc work, ~1 hour, no CPU]
- **Point:** CRITIQUE_EXAMINER measured the missing naive-randomisation rows (uniform-stack
  0.442 on 35-159; uniform-independent 0.546) and menu-sufficiency (equilibrium stable from R=8)
  with `scratch/uniform_stack_probe.py`, but the numbers live only in that critique file.
- **Goal:** the gen13/gen14 ladders and any thesis figure carry the uniform-stack and
  uniform-independent rows; one menu-sufficiency sentence lands in the gen13 ledger. This converts
  "SACRED beats ALNS" (structurally cheap) into "SACRED beats every uncalibrated strategy class"
  (the real claim).

### A1. The LP-distillation control [TRAINING, ~1-2 days; the single most important item]
- **Point:** the ZST act shows adversarial training transfers (gen16: held-out Gdansk 1.68x its
  equilibria). An ML examiner will ask: could you not just TRAIN THE SAME NETWORK TO IMITATE THE
  SOLVER? Labels are free at K=1 (the LP solves in milliseconds). Until this control runs, the
  flagship claim is exposed to a one-line rebuttal.
- **Goal:** train the gen16 architecture on the gen16 multi-city pool with a supervised loss (KL
  to each training instance's oracle equilibrium occupancy), no adversary; evaluate zero-shot on
  the same held-out Gdansk and Istanbul ODs under the identical ratio metric, with dual selection
  reporting.
- **Outcome branches (all citable, pre-register them):** SACRED beats distillation = adversarial
  interaction contributes beyond label-fitting. They tie = SACRED is an equilibrium amortiser
  that needs no labels (still a claim, plus labels stop existing past the enumeration wall at
  K>=4 where only self-play can train). Distillation wins = the ZST act is re-scoped honestly
  before an examiner does it.
- **How:** `scripts/train_generalist.py` is the template; the oracle labels come from
  `solve_multiconvoy`. Design details: CRITIQUE_EXAMINER §4.2 and Tier 1 item 2.

### A2. The shuffled-map transfer row [eval-only, ~half a day; equal-first in importance]
- **Point:** every threat map any policy has EVER trained on is an affine transform of edge
  length (`length_band_vulnerability`), so "the policy conditions on the threat map" is currently
  indistinguishable from "the policy conditions on road geometry". Measured basis:
  `scratch/threatmap_geometry_probe.py` (route cost vs route worst-vulnerability |corr|
  0.60-0.99 on 8/8 pool instances; permuted maps shift the equilibrium strategy L1 0.44-1.03).
- **Goal:** take the FROZEN gen16 generalist; on the held-out Gdansk ODs, randomly permute each
  instance's vulnerability values across its candidate edges; recompute each shuffled instance's
  oracle equilibrium and best response; feed the policy the SHUFFLED map; report the
  ratio-to-equilibrium beside gen16's 1.68 and a random-init reference.
- **Outcome branches:** policy tracks the shuffled equilibria = the map-conditioning claim is
  earned and STRONGER than currently written (transfer across threat fields, not just cities).
  Policy collapses toward random = the claim is reworded to "geometry-conditioned under a
  geometry-consistent threat model", and the recorded fix is a randomised-map training pool
  (one gen16-scale retrain, Kilian's go). Full reasoning: CRITIQUE_12-07-26.md §3.1.

### A3. The intel-noise robustness curve [eval-only, ~half a day]
- **Point:** every trained defender has always observed the TRUE threat map. Operationally,
  threat intelligence is noisy, and this is the first question a defence-domain examiner asks.
- **Goal:** evaluate the frozen generalist under perturbed OBSERVED maps (multiplicative noise
  and/or top-k dangerous-edge dropout) while scoring against the TRUE map's oracle best response;
  plot degradation vs noise level with true-map policy and random-init as anchors. Graceful
  degradation = a robustness exhibit; a cliff = a scope sentence. A2's shuffled map is this
  curve's extreme point; build them on one harness. Reference: CRITIQUE_12-07-26.md §3.3.

### A4. Complete the ZST causal chain: gen21 to n=3 + one DR-generalist control [TRAINING, ~half a day of machine time, background-sized]
- **Point:** "adversarial training is causal for transfer" currently rests on ONE vanilla seed
  (gen21), and the vanilla control changes BOTH the adversary and the objective at once, so
  "best-response pressure" and "any threat exposure" are not yet separated.
- **Goal:** (a) two more gen21 vanilla-generalist seeds (upgrades the causal sentence to n=3);
  (b) ONE domain-randomisation generalist: same mission objective, interdictor sampled uniformly
  at random each sortie (threat exposure WITHOUT best-response pressure), evaluated zero-shot on
  the same held-out ODs. DR much worse than smooth-FP training = the claim sharpens to
  "best-response pressure is causal". DR ties = the honest wording is "threat-aware training".
  Reference: CRITIQUE_12-07-26.md §4.1; CRITIQUE_EXAMINER §4.3.

### A5. d3_gdansk reliability check [eval-only, hours]
- **Point:** the poster-intended claim ("on an unseen city, designing against the deployed
  policy is almost uncorrelated with designing against the equilibrium", 0.109 vs 0.768) is
  ambiguous between signal and noise: nobody has checked whether the policy-exploitability
  target on Gdansk designs is stable under re-evaluation.
- **Goal:** re-evaluate the Gdansk design sweep with fresh evaluation seeds; report the
  test-retest correlation of the target and the disattenuated policy-vs-oracle correlation. The
  poster claim is GATED on this: reliable target + still-low correlation = the claim is earned;
  unreliable target = the exhibit moves to "in-distribution only". Reference: CRITIQUE_EXAMINER §4.4.

### A6. The retrieval baseline for the ZST act [eval-only, hours]
- **Point:** a trivial amortiser might match the generalist: find the most similar TRAINING
  instance (by the F3 feature vector), play ITS equilibrium mixture mapped onto the new menu by
  route rank. If naive retrieval ties the generalist, the GNN adds little; if the generalist wins
  clearly, the claim strengthens. Complements A1 (retrieval = memory without generalisation;
  distillation = labels without interaction; SACRED = interaction).
- **Goal:** one table row per held-out city beside the gen16 numbers. Reference:
  CRITIQUE_12-07-26.md §5.4.

### A7. Gap-closure restatement + the transfer-decay figure [eval-only, ~half a day]
- **Point:** the ratio-to-equilibrium metric flatters cells where the deterministic optimum is
  already close to the equilibrium. Restated as the fraction of the
  deterministic-to-equilibrium gap CLOSED, the transfer ladder decays roughly 90% (trained
  instance) -> ~56% (held-out OD) -> ~52% (Gdansk) -> ~24% (Istanbul) -> ~8% (whole Kyiv)
  (estimates from screen medians). The thesis must show this before an examiner derives it.
- **Goal:** recompute gap closure per OD exactly from the saved JSONs for every transfer cell;
  produce the decay-vs-transfer-distance figure; fold one honest sentence into each affected
  ledger. This is simultaneously more honest AND a better figure than threshold passes.
  Reference: CRITIQUE_12-07-26.md §3.2.

### A8. The prevalence figure [oracle-only, ~half a day, free]
- **Point:** answers "did you cherry-pick your instances?" for the whole thesis in one figure.
- **Goal:** over every high-connectivity OD pair in all four cities, compute uniform-stack/
  equilibrium and loss_det/equilibrium ratios; plot the distributions; mark the headline
  instances on them. Shows how OFTEN calibrated mixing matters and where the headline instances
  sit in that population. Reference: CRITIQUE_EXAMINER §3.8 + Tier 1 item 5.

---

## Block B: the differentiators (after Block A; each is one bounded, pre-registered ledger)

### B1. The holistic-SBO integration-gap experiment [mostly eval, ~1-2 days; do BEFORE B2]
- **Point:** Objective 4's verbatim promise is "holistic, SIMULTANEOUS evaluation of strategic
  supply chain design alongside the operations-level SDVRP". The current D-chain optimises the
  tiers SEPARATELY, so the word "simultaneous" is not yet earned. This is the experiment that
  earns it, and it is Kilian's named direction (unified strategic/tactical/operational planning).
- **Goal:** joint design space = base placement x hardening allocation x fleet size. Arm A
  optimises tier-by-tier (the classical decomposition: D1 then D2 then N); arm B runs ONE
  surrogate-guided loop over the joint space at the SAME total evaluation budget. Both priced by
  the frozen generalist's exploitability AND fleet cost (report a bi-objective frontier, not a
  point). Primary = the integration gap (joint minus sequential), with D2's measured tier
  coupling (equilibrium mass shift L1 = 0.29) as the mechanism story. Run it on Gdansk and it is
  simultaneously the strongest Obj-4 sentence and the poster exhibit. A gap of ~0 is also a
  finding (the tiers decompose; say why). References: CRITIQUE_EXAMINER §6 item 8;
  CRITIQUE_12-07-26.md §6 item 6.
- **Why before B2:** it buys thesis marks directly (an objective's verbatim wording); B2 buys
  publicity and a spin-out. Both should run; this one runs first.

### B2. The agentic-LLM exploitability benchmark [eval-only, ~1-2 days]
- **Point:** Kilian's named idea. The security game has a computable optimum, a computable
  deterministic trap, and a graded ladder between them, which makes it an unusually clean
  benchmark for language agents; nobody has scored LLM planners on a security-game yardstick.
  It also independently supports the thesis mechanism: calibrated randomisation is exactly what
  language agents lack unaided.
- **Goal:** 2-3 frontier LLMs, three pre-registered registers: (a) deterministic ("choose a
  route"; expect ~loss_det), (b) stated-strategy ("output a probability distribution over the
  menu"; score the stated mixture exactly under the oracle BR), (c) agentic-sequential (T sorties
  with interception feedback, optionally vs the gen19 pattern-of-life adversary where the dynamic
  optimum 0.049 is computable: does in-context adaptation discover anti-repeat hedging?).
- **Design decision to make UP FRONT: tool use.** With code execution a frontier model will just
  solve the LP and land at equilibrium; the informative registers are no-tools, with
  tools-allowed reported as a separate ceiling row. Pin model versions, temperature, prompts; log
  all transcripts as the reproducibility record.
- **Warning (agreed):** in a 12,000-word thesis this earns AT MOST one subsection and one ladder
  column; its natural home is a workshop-paper spin-out. If it threatens the writing calendar it
  moves whole to post-freeze. References: CRITIQUE_EXAMINER §6 item 7; CRITIQUE_12-07-26.md §6
  item 5.

### B3. The risk-aversion spectrum [oracle-only, ~half a day, free]
- **Point:** the objective-selection story ("we use the loss-averse mission objective because
  that is where randomisation pays") is currently a defended binary choice. One oracle sweep
  turns it into a measured LAW and immunises against "you picked the objective where you win".
- **Goal:** sweep the loss-aversion parameter (m-of-N / CVaR-style weighting; the oracle already
  supports threshold objectives) and plot the deterministic-vs-mixed gap against it: "the price
  of predictability as a function of loss-aversion". Reference: CRITIQUE_EXAMINER §6 item 9.

### B4. The multi-OD correlation-gap probe [oracle-only, free]
- **Point:** all convoys currently share one origin-destination pair. The post-thesis game that
  would fix this (different destinations sharing corridor edges) is only worth building if
  correlated joint routing beats independent routing by a real margin.
- **Goal:** extend the oracle to joint route tuples for 2-3 convoys with different destinations;
  measure the correlation gap (best correlated joint mixture vs best product of independents).
  Material gap = the Tier-3 game is justified; no gap = a clean scoping result for free.
  Reference: CRITIQUE_EXAMINER §6 item 10.

---

## Block C: deliverables and writing (moved to the BACK by Kilian's 2026-07-12 decision)

### C1. Chronicle + doc hygiene [doc work, hours; do FIRST within Block C]
- **Point:** `SACRED_PROGRESS.md` ends at entry 21 (through gen19); gen20-gen23 and the
  expansion completion have no entries, and the HANDOVER banner stack is seven layers deep. The
  FAR (C2) is written FROM the chronicle, and future agents onboard from these documents.
- **Goal:** append chronicle entries for gen20-23 and every Block A/B item completed by then;
  one consolidation pass on the HANDOVER banners; ledgers remain the sole number source.

### C2. Final Activities Report + presentation [days; HARD EXTERNAL DEADLINE 30 JULY 2026]
- **Point:** a compulsory course deliverable (maximum 2 pages + a presentation to the
  supervisor). Moved to the back of the ORDERING, but the DATE does not move: whatever state the
  programme is in, this must be drafted and delivered before 30 July. Plan backwards from that.
- **Goal:** summarise work done and what remains, drafted from the chronicle; keep claims at
  ledger strength (no "all objectives met" shorthand; use the per-objective deltas from the two
  2026-07-12 critiques §1).

### C3. The interactive exhibit [presentation work, ~half a day]
- **Point:** Objective 2 promises a "visual, interactive" environment and none of the
  interdiction-era results are interactive; 20% of the thesis mark is structure/presentation,
  and the poster needs a centrepiece.
- **Goal:** an HTML map of a held-out city where the reader toggles
  shortest-path / ALNS / uniform-stack / SACRED and sees route distributions and interception
  numbers. All numbers already exist; this is presentation only. First item to drop if time
  compresses.

### C4. The storyline rewrite + hostile self-critique [writing, ~2-3 days]
- **Point:** nineteen-plus generations must fit 12,000 words with one spine. This is
  `NEXT_STEPS_11-07-26.md` item 7, kept verbatim but moved behind the computational work.
- **Goal:** rewrite `THESIS_STORYLINE.md` onto the four-act spine (CRITIQUE_EXPANSION §4.7):
  (I) the negative campaign, compressed; (II) the security game with the two post-fix ladders;
  (III) ZST + the SBO stack as the payoff; (IV) the measured boundaries as the discussion.
  Fold in every Block A/B result; score the five objectives honestly with deltas named; write
  the ZST-vs-LP paragraph and the gen19 quantal-response framing explicitly. Then interrogate
  the rewritten storyline as a hostile examiner (both 2026-07-12 critiques §1-§5 are the
  checklist), correct autonomously where cheap, and stop after two rounds.

### C5. Freeze and write [after 3 August: writing only]
- Freeze the repo state on 3 August (tag it); thesis and poster writing to 28 August. The
  thesis-planner brief lives at `../../thesis/THESIS_PLANNER_HANDOFF.md`. Nothing in the
  post-freeze register below starts before submission.

---

## Drop order (if the calendar bites)

Drop from the FRONT of this list first: C3 (exhibit) -> B2 (LLM benchmark; moves whole to a
post-freeze spin-out) -> A4's DR arm (keep gen21 n=3) -> A6 (retrieval baseline) -> B4 -> B3.
**Never drop:** A1 (distillation), A2 (shuffled map), A7/A8 (gap-closure + prevalence figures),
C1, C2 (external deadline), C4. Those decide what the thesis is allowed to claim and whether the
30 July deliverable exists.

## Post-freeze / post-submission register (recorded, NOT scheduled)

PSRO/double-oracle population training; optimistic/extragradient last-iterate dynamics (the
known fix family for the transient finding; deliberately out of thesis scope); the multi-OD
interdiction game (gated on B4's probe); the A4 K=5 trained cell (stays deferred, per the
2026-07-11 decision); the "contested-logistics gym" benchmark release; full B1 (Poisson-demand
campaign) and B5 (deception/decoys); a history-aware generalist (gen19's window conditioning
inside the gen16 multi-city recipe). Publication map: CRITIQUE_EXAMINER §6 end; run Block A
BEFORE drafting any abstract, because A1/A2 decide which ZST story any paper can tell.

---

## Repo state at handover (2026-07-12; refreshed 2026-07-15)

Branch `gen08-interdiction`, suite green, nothing running. **As of 2026-07-15 the working tree is
CLEAN at `5cd1e02`** (the previously-untracked `CRITIQUE_EXAMINER.md`, `CRITIQUE_12-07-26.md`,
`scratch/uniform_stack_probe.py`, `scratch/threatmap_geometry_probe.py`, and this file were all
committed during the 2026-07-13 run; there is nothing outstanding to commit). Blocks A + B are
complete bar B2-live (API keys); Block C is not started, paused on Kilian's instruction.
Persistent agent memory lives at
`~/.claude/projects/-Users-kilian-Kilian-ICL-Thesis-code-sacred/memory/` (read `MEMORY.md` at
session start; keep it and the chronicle current as items complete).
