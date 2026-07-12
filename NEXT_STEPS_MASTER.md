# NEXT_STEPS_MASTER.md: the agreed work programme after 2026-07-12 (checklist for incoming agents)

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
> - [~] **A4** RUNNING (`experiments/gen25_dr_control.md`): vanilla seeds 1-2 + DR seed 0,
>   12000 sorties each, in flight.
> - [x] **A5** DONE (`experiments/d3_gdansk.md` appendix): the 0.109 was seed-0-specific
>   (seeds 1/2: 0.443/0.433); cross-seed reliability 0.32-0.54; poster claim DOWNGRADED.
> - [x] **A6** DONE (`experiments/a6_a7_a8_completions.md`): retrieval MATCHES the generalist
>   (1.676 vs 1.677/1.733); transfer ladder fully bounded.
> - [x] **A7** DONE (same ledger + `assets/transfer_gap_closure.png`): gap-closure ladder
>   0.90 -> 0.54 -> 0.45 -> 0.20 (Istanbul) -> 0.04 (Kyiv); far-end wording rule recorded.
> - [x] **A8** DONE (same ledger + `assets/prevalence.png`): 69% of 160 ODs have det/eq >= 2;
>   headlines sit in the top decile BY SCREEN DESIGN.
> - [~] **B1** RUNNING (`experiments/b1_integration_gap.md`): joint-vs-sequential design gap,
>   two actors, in flight.
> - [!] **B2** HARNESS READY (`experiments/b2_llm_benchmark.md`, dry-run validated): live runs
>   BLOCKED on API keys + spend cap from Kilian.
> - [x] **B3** DONE (`experiments/b3_b4_oracle.md`): three-regime risk-aversion law (mission =
>   the unique objective determinism cannot escape by spreading).
> - [x] **B4** DONE (same ledger): multi-OD correlation gap median 14.4% (bar met; Tier-3
>   multi-OD game justified).
> - Block C: NOT STARTED (paused per Kilian's instruction; the FAR's 30 July external deadline
>   stands regardless).

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

## Repo state at handover (2026-07-12)

Branch `gen08-interdiction`, suite 161 green, nothing running. Untracked at the time of writing
(commit them so future agents see them): `CRITIQUE_EXAMINER.md`, `CRITIQUE_12-07-26.md`,
`scratch/uniform_stack_probe.py`, `scratch/threatmap_geometry_probe.py`, and this file.
Persistent agent memory lives at
`~/.claude/projects/-Users-kilian-Kilian-ICL-Thesis-code-sacred/memory/` (read `MEMORY.md` at
session start; keep it and the chronicle current as items complete).
