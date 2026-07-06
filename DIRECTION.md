# DIRECTION.md: the contested-resupply redirection (opened 2026-07-06)

> **STATUS: agreed with Kilian 2026-07-06 · SUPERVISOR SIGN-OFF PENDING · nothing built, no CPU
> spent on this direction yet.** The gen03-gen06 campaign record (`HANDOVER.md`,
> `SACRED_PROGRESS.md`, the `experiments/genNN_*.md` ledgers) stands unchanged and is the evidence
> base for everything here. This file records the **new view and the new direction**. The active
> plan lives in `ROADMAP.md`; the thesis-facing argument lives in `THESIS_STORYLINE.md`. House
> rules are unchanged (`SYSTEM.md`): citable numbers come from ledgers only; plan first, never
> dive in; Kilian owns CPU spend, design changes and launches; consult him when unsure.

## 0. TL;DR for whoever picks this up

The completed campaign established, with pre-registered rigour, that adversarial co-training as
naturally formulated does **not** confer average-case robustness in this problem class, and
measurably worsens it (gen06 ledger: pooled dD_targeted = −881 ± 284, 0/3 pairings positive;
robustness ranking greedy > vanilla > adversarially-trained). On 2026-07-06 Kilian posed the
carte-blanche question: with anything changeable **except SAC, the protagonist/antagonist
dynamic, and RL**, can adversarial training be made to show a real benefit? The converged answer:
**change the question to the one minimax training actually answers.** Adversarial training's
native product is *worst-case* performance against a *strategic, adaptive* opponent, measured as
**exploitability**, not average degradation under fixed disruptions. Two of the four alternative
applications in Kilian's 2026-07-05 supervisor deck are exactly that game; with Kilian we chose
**contested autonomous resupply** (application 1) as the working frame.

New headline claim to demonstrate (gen07): *adversarial co-training buys calibrated
unpredictability: policies that sit on a better efficiency-versus-exploitability frontier than
deterministic dispatchers (greedy, rolling-ALNS, near-deterministic vanilla SAC) and than naive
randomness.* Five engineering fixes, one per diagnosed campaign pathology, make both agents
learnable. Honest success estimates: **~75-85%** for the exploitability headline (Tier 1),
**~40-60%** for the old average-case metric (Tier 2), which we keep as the secondary and report
either way.

## 1. Where this came from

1. **gen06 closed the old question.** See `HANDOVER.md` §1 and the gen06 ledger. The negative is
   real, competence-gated, and consistent with the literature's reactive-dominance finding
   (Ritzinger et al. 2015, cited in the literature review).
2. **Kilian's carte-blanche brief (2026-07-06, this session):** total freedom to make SACRED
   work, defined as "adversarial training shows a real benefit", with three invariants: SAC stays,
   protagonist/antagonist stays, RL stays.
3. **Supervisor context:** Dr. Angeloudis wants to look at different applications. Kilian's deck
   `../../Weekly Presentations/06.07..pptx` lists four; he also pointed at Panopticon AI and the
   AAMAS conference (see §8).

| # | Application (from the deck) | Adversary type | Verdict for the headline |
|---|---|---|---|
| 1 | **Tactical autonomous swarm logistics in contested environments** (drone resupply of forward operating bases; enemy AA/EW jams flight corridors) | Strategic, adaptive | **CHOSEN.** Mechanically closest to the existing stack (corridor denial = full-block antagonist; FOB resupply = Poisson stream) |
| 2 | Post-disaster humanitarian logistics (cascading arc failures) | Nature (non-strategic) | Motivation/framing only; average-case regime where gen06's negative applies; DR is the honest comparator |
| 3 | High-value asset escort in hostile urban theatres (interception) | Strategic, adaptive | **Stretch variant** (cleanest theory: sparse, attributable interception reward); recorded, not scheduled |
| 4 | LEO constellations / Kessler syndrome | Nature | Out of scope (new 3D environment; no time) |

The deck's own slide 1 lists "High-Entropy Routing" as a defining SACRED characteristic and
slide 3 says "unpredictable and untargetable edges are resilient to interception": the reframe
below is the formalisation of Kilian's own intuition.

## 2. The central insight: two registers of robustness

**Register 1: disruption robustness (the campaign's).** Average degradation under fixed, held-out
attacks. The campaign showed reactive re-planning captures nearly all value here (greedy won the
gen06 ranking), matching Ritzinger's reactive-dominance. Refighting this fight, even with all
fixes, is a coin flip.

**Register 2: worst-case against a strategic adversary (the redirection's).** Measured as
**exploitability**: *how badly an enemy who has learned your habits can hurt you.* Operationally:
freeze the trained dispatcher; give the attacker a fixed budget to prepare the best attack
against **that specific dispatcher** (train a best-response attacker AND fit the scripted attack
family to it); exploitability = the damage of the strongest of those, measured as D on paired
instances. The key distinction is knowing the *strategy* versus knowing *today's action*:
over repeated sorties the strategy always leaks (pattern-of-life analysis), the per-sortie coin
flips do not. A deterministic policy is fully predictable, hence ambushable with near-certainty;
a well-mixed policy denies the ambush its expected value even when the mixing probabilities are
public. In game-theoretic terms exploitability is the best-response gap, the distance from the
minimax solution; minimax (adversarial) training reduces it *by construction*. That is why Tier 1
is winnable where the campaign's question was not.

**The claim is a frontier, not a point.** Anyone can be unexploitable by being uniformly random,
at ruinous clean cost. The scientific claim is that adversarial co-training finds *cheap*
unpredictability: it mixes only where interception risk justifies the detour cost. Controls are
designed to isolate exactly this (entropy-matched vanilla = "not just noise"; DR-SAC = "not just
any attacked training").

**In-house evidence that this is the right axis** (all ledgered): gen05 BR rows: against the
competent, deterministic greedy the learned attacker became the portfolio's strongest attack
(+1667 vs scripted's +1154/+714); gen06 BR rows: br_scripted hits greedy for 1715; and the gen06
arms' entropy telemetry (§4) shows vanilla SAC anneals toward near-determinism, i.e. toward
exploitability.

## 3. The new game: contested resupply

Skin, not surgery. The mapping from the existing stack:

| Existing (SDVRP last-mile) | Contested resupply reading |
|---|---|
| Kaliningrad graph / chokepoint arenas | Contested theatre; corridors and gateways |
| Depots (110/135) | Logistics hubs / FOB cluster anchors |
| Poisson request stream | Resupply demand from forward positions |
| Trucks, capacity-1 | Autonomous resupply vehicles/drones |
| Full-block congestion antagonist, budget | Corridor denial (EW jamming / area denial), sortie budget |
| Latency reward (−queue/tick) | Total resupply delay |
| Greedy insertion dispatcher | Doctrine baseline: deterministic reactive dispatch |

What stays: the physics engine, SMDP wrapper, GATv2 nets, SAC core, ATLA alternation, ledgers and
the whole evaluation discipline (pre-registration, competence gates, paired instances, stochastic
eval, validation/test splits). What changes: the headline metric (§2), the five fixes (§4), the
curriculum/exposure machinery, and the arm set (§5). Variant B (interception physics: ambush
events with catastrophic cost instead of delay, per application 3) is recorded as the stretch
design if the resupply variant under-delivers; it is theoretically cleaner (perfectly
attributable sparse rewards) but needs ~a week of new build.

## 4. Five diagnosed pathologies → five fixes

The campaign's failure mechanisms, each with its fix and its anchor. This table is the design
contract for the build phase (`ROADMAP.md` Phase B).

| # | Pathology (evidence) | Fix | Anchor |
|---|---|---|---|
| 1 | Overpowered constant attack → collapse-regime training (gen06 arms trained at delivery 0.18-0.27 vs vanilla 0.66; queue ~2x; see telemetry note below) | **Attack-strength + exposure curriculum**: mixed clean/attacked episodes, budget ramps gated on the defender staying inside a competence band | Curriculum Adversarial Training (IEEE 9892908); QARL bounded-rationality curricula (arXiv 2311.01642); RARL's own overpowered-adversary caveat |
| 2 | Zero-sum latency reward buries controllable signal under the shared queue baseline (gen03/04/06 SNR theme) | **Counterfactual/difference rewards**: per-episode clean-twin rollout with common random numbers; attacker reward = damage above twin; defender reward = return minus twin baseline. Note: the subtracted term depends on neither agent's in-episode actions, so the game stays zero-sum up to a per-episode constant and the equilibrium is preserved | Difference rewards / counterfactual baselines (COMA lineage) |
| 3 | Entropy machinery mis-calibrated: the 0.45·ln(N) target scales with the attack-inflated backlog; temperature never anneals under attack | **Entropy repair**: fixed absolute per-decision-type targets; lower antagonist target (the gen04b hypothesis); defender entropy becomes a tuned, reported quantity (it is the product in this game) | Own diagnosis (gen04 ledger; telemetry below) |
| 4 | The learned adversary cannot learn in a flat ~120-option action space (gen03/04: below random; entropy pinning) | **Attacker learnability package**: factored action (pick asset, then edge on its route), route-reach mask, motion features (built, gen04 N1), counterfactual reward, sane entropy target; plus an **adversary population** (scripted seeds + successive learned BRs; defender trains against the mixture) to kill co-evolution cycling | gen05's +1667 existence proof; PSRO/double-oracle and fictitious self-play (the latter already cited in the literature review) |
| 5 | γ = 0.99/tick myopia vs 100+-tick consequences | γ 0.997+ or per-event discounting emphasis, n-step targets | Standard |

**Telemetry note (2026-07-06; reproduced and ledgered per ROADMAP A3.1).** A comparative read of
the gen06 tfevents (`logs/tb_runs/gen06_dynassign_matrix`, windowed means; committed probe
`scratch/gen06_telemetry_probe.py`, results in the gen06 ledger's post-hoc appendix) found
systematic arm differences that the gen06 ledger's original mechanism paragraph does not capture:
final SAC alpha 0.13 (vanilla, all seeds) vs 0.62-0.86 (scripted arms); final policy entropy
0.37-0.39 vs 0.47-0.52; training-time delivery 0.66 vs 0.18-0.27 and final queue ~17 vs ~35-40;
protagonist Q_Spread 2.6-3.8 vs 13-15 (HIGHER under attack, contradicting a naive "critic cannot
discriminate" reading for the protagonist; that reading belongs to the gen03/04 antagonist only);
critic loss ~200-225 vs ~870-1130; clean eval curves flat from ~ep 50 in all six arms. Three
mechanism candidates are recorded: M1 reward SNR (as ledgered), M2 entropy-target mis-scaling
with backlog, M3 collapse-regime state distribution. The A3.2 (robustness-vs-training-time) and
A3.3 (matched-temperature) probes discriminate between them; their results live in the same
gen06 ledger appendix.

**Probe outcomes (2026-07-06, overnight; details in the gen06 ledger appendix).** A3.3 rules
out evaluation-time temperature as the cause of the gen06 gap: at matched determinism the
reversal persists and even widens (pooled dD_targeted −1284 ± 310 at tau 0.5, −956 ± 370 at
argmax; tau 1.0 reproduces the ledgered −881 ± 284 exactly). The deficit is in the learned
policy; M2 stays relevant only as a training-time channel, M1+M3 carry the mechanism. A3.2 adds
the sharpest new fact: **vanilla's aimed-attack robustness DECLINES with clean training time**
(5/6 run-attack cells worse at ep650-800 than ep50-200, up to +14%), while the scripted arms
start worse and only slowly recover. Clean-task specialisation buys competence at the price of
predictability, in the campaign's own data: direct in-house support for the exploitability
register (§2) and for the frontier claim (calibrated, not incidental, unpredictability). A3.4:
the gen06 reversal is significant under the pre-registered pooled criterion with 3/3 sign
consistency, but the conservative 3-pairing t-CI includes zero; the thesis wording carries both
levels.

## 5. The gen07 programme (concept level)

Exact metrics, estimators and thresholds get **fixed in `experiments/gen07_*.md` before looking**,
per house rule. Concept:

- **Arms.** Core: `vanilla` (no adversary) vs `sacred` (curriculum ATLA vs the adversary
  population). Controls (wave 2): `dr` (random-attack training) and `entropy-matched vanilla`
  (isolates "just add noise"). Optional (Obj-3 ablation): ERB-seeded variants. References
  (no training): `greedy`; `rolling-ALNS` dispatcher (eval-only build) if funded.
- **Seeds:** ≥3 per arm; paired instances; stochastic protagonist eval; selection on validation
  attackers only.
- **Primary (Tier 1): the exploitability gap.** For each arm, Expl(arm) = the strongest attack
  in that arm's tailored portfolio (learned BR trained against it + scripted attacks fitted to
  it), as degradation D on paired test instances. Success concept: Expl(vanilla) − Expl(sacred)
  > 0 with the pre-registered CI excluding 0 and sign-consistency across seed pairings; secondary
  headline: sacred less exploitable than the deterministic references at bounded clean premium.
  The portfolio-max rule means the metric never depends on a possibly-weak learned BR alone
  (gen03/04 lesson).
- **Secondary (Tier 2):** the gen06-style held-out attack portfolio (continuity with the
  campaign), clean premium, and the efficiency-versus-exploitability frontier plot; budget-axis
  sweeps reported as curves, not points.
- **Statistical reporting rule (new, learned from gen05/gen06):** report the pre-registered
  instance-level pooled CI AND per-pairing sign consistency AND the seed-level sensitivity
  analysis. Never claim "significant" from one level alone.
- **Gates before any long run** (all cheap, all pre-registered): suite green; timing probe (both
  phases); competence probe on the contested arena; **the gen04 gate re-run with the learnability
  package** (a retrained BR attacker must beat random blocking). If the BR gate still fails, the
  exploitability metric survives via the fitted-scripted portfolio, and the BR failure itself is
  a reportable finding.

## 6. Research objectives under the redirection

Full argument in `THESIS_STORYLINE.md`. Summary:

| Objective | Prognosis | Mechanism |
|---|---|---|
| 1. Asymmetric zero-sum Markov game | Met, deepened | Same game; diagnosed learnability conditions; exploitability = measured distance from the game's solution concept |
| 2. Simulation environment | Already met | Existing env + visualiser; contested skin is a relabelling plus curriculum module |
| 3. SAC + ATLA + ERB bootstrapping | Met with one honest adaptation | ATLA population training; ERB demo-seeding as a pre-registered time-to-competence ablation (demos from the dynamic dispatcher; note the "population-based" taxonomy adaptation openly) |
| 4. SBO (depots/fleet) | Reduced-form demonstration, or descope | Neural surrogate over a depot-placement grid predicting performance AND exploitability; validated against full simulation; eval-only |
| 5. Evaluation vs metaheuristics + vanilla SAC under varied disruption | Met substantively | Vanilla/DR/entropy-matched controls; budget sweeps as curves; rolling-ALNS as the SOTA-adaptive reference, whose determinism makes it measurably exploitable (a finding, not a formality) |
| ZST (from the aim) | One cheap test | Zero-shot transfer of final arms + attack portfolio to a held-out geometry; eval-only |

## 7. Honest risk register

- **Tier-1 residual risks:** vanilla's leftover entropy may shrink the gap vs vanilla (hence the
  entropy-matched control and the deterministic references, where the gap should be large);
  BR-attacker quality (mitigated by the portfolio-max rule and the learnability package).
- **Tier 2 may stay null.** Pre-registered as secondary; a second null is reportable and the
  thesis narrative survives it ("worst-case benefit demonstrated; average-case transfer remains
  negative, and here is why").
- **Calendar.** This blows the old ~Jul 16-18 freeze. Proposed new experimental freeze
  ~Aug 3-7 against the hard thesis deadline (10:00, 28 Aug 2026, 12,000 words + poster). Writing
  runs in parallel from now (thesis planner). If anything slips, the fallback is the two-act
  freeze-and-write on gen06, which was already judged defensible.
- **Examiner attacks to pre-empt** (bank maintained in `THESIS_STORYLINE.md`): "just add noise"
  (entropy-matched control); "your BR attacker defines the metric" (portfolio-max); "you changed
  the question after losing" (pre-registered redirection, recorded here BEFORE gen07 was built,
  with the campaign's negative published in full as act one); seed-level vs instance-level
  significance (dual reporting rule); "zero-sum with a baseline is not zero-sum" (constant
  per-episode term; equilibrium preserved).

## 8. External context (recorded 2026-07-06)

- **Panopticon AI** (https://panopticon-ai.com): defence-AI company; **BLADE** web platform for
  military simulation, Gymnasium-compatible, Apache 2.0, explicitly for training RL in wargaming
  scenarios (AlphaStar-inspired). Relevance: industry validation of the contested-logistics
  framing; a platform, not competing research; a plausible post-thesis demo target. Not touched
  before the deadline.
- **AAMAS** (supervisor's pointer): the home venue of security games, randomised patrolling,
  PSRO/empirical game theory and adversarial MARL, i.e. exactly the community whose currency is
  exploitability. AAMAS 2027: 3-7 May 2027, Hanoi; submissions expected Oct 2026 (TBC per the
  Warwick AAMAS 2027 page). **Parked (Kilian 2026-07-06: no conference thinking now);** kept
  here only as recorded context because the supervisor raised the venue. The research-community
  fit still informs the literature the thesis cites (security games, patrolling, PSRO).

## 9. Decision gates pending (Kilian owns all; supervisor where marked)

> **Decision log 2026-07-06 (morning round 2, Kilian: the A1 agenda resolved directly):**
> (1) **Reframe ADOPTED** (exploitability headline, three-act thesis). (2) **Applications:** per
> recommendation: contested autonomous resupply leads; humanitarian logistics as civilian
> motivation; asset escort recorded as stretch. (3) **Freeze: Aug 3, HARD.** (4) **Obj 4:
> reduced-form demonstrator** (robust depot placement surrogate). (5) **Rolling-ALNS arm:
> funded** (eval-only). (6) **ERB bootstrapping: include, modestly scoped** ("at least a
> little": the wave-2 time-to-competence ablation, first to drop if the calendar bites).
> (7) **ZST: one held-out-geometry transfer test, confirmed.** Kilian also authorised the
> **Phase-B build start immediately** (ahead of the supervisor conversation; fallback to
> freeze-on-gen06 unchanged if that conversation redirects). Status banner above remains
> accurate: supervisor sign-off itself is still pending as an event; the working decisions are
> now Kilian's own.
>
> **Decision log 2026-07-06 (build round, Kilian):** B1 reward baseline = **Option B**
> (greedy no-attack twin; directly targets the M1 SNR pathology). B4 population = **B4-lite**
> (fixed scripted-attacker mixture; B4-full/PSRO stays a recorded stretch). **Environment/graph
> is UNCHANGED** from gen02-gen06 (same Kaliningrad OSM graph, depots 110/135, hotspot band,
> Poisson arrivals): the contested arena differs from dynassign only by `antag_reach="route"`,
> a masking rule, so gen07 is directly comparable to the campaign in the same measured
> environment. The only deliberate other-geometry uses are the scoped ZST transfer test and the
> Obj-4 depot-placement grid (both eval-only, both later). Build proceeds; **training NOT
> launched** (Kilian will greenlight).
>
> **Decision log 2026-07-06 (evening, Kilian):** (1) git commit approved; separation required
> for framework changes → `main` frozen for `src/` from now, gen07 code on a dedicated branch
> (policy recorded in `ROADMAP.md` Phase B). (2) A3 evidence-hardening probes: all four
> approved, executed autonomously overnight. (3) Conference/publication topics (AAMAS etc.)
> dropped from all current planning per Kilian's instruction; the gen07 ledger draft (A4) is
> held for explicit confirmation 2026-07-07 morning (it is the experiments ledger, not a
> conference artefact). (4) Thesis planner launch: not now. (5) Phase-B start: decided
> 2026-07-07 morning after reviewing the A3 outputs.

1. **Supervisor sign-off** on: the exploitability reframe; the contested-resupply skin; the new
   freeze date; Obj-3 ERB ablation wording; Obj-4 reduced form vs descope; rolling-ALNS arm;
   AAMAS ambition. Materials: this file + `THESIS_STORYLINE.md`.
2. **Funding decisions** (CPU/build time): wave-2 controls, ERB ablation, rolling-ALNS baseline,
   Obj-4 surrogate demo, ZST test. Each is itemised with cost in `ROADMAP.md`.
3. **Thesis planner launch** (`../../thesis/THESIS_PLANNER_HANDOFF.md` needs a redirection
   banner before launch; see `ROADMAP.md` A2).

## 10. Doc web (state of record after this redirection)

- `DIRECTION.md` (this file): the current view + direction. **Living.**
- `ROADMAP.md`: the active plan (short/mid/long term). **Living.**
- `THESIS_STORYLINE.md`: the thesis argument, objective by objective. **Living.**
- `HANDOVER.md`, `SACRED_PROGRESS.md`, `CRITIQUE.md`, `experiments/*`: the campaign record.
  Historical + append-only (PROGRESS gains entries as gen07 runs).
- `CONTEXT.md`, `PROBLEM_REDESIGN.md`, `TASK.md`: historical (banners say so).
- `SYSTEM.md`: operating dogmas. Living, unchanged by this redirection except the §5 epic state.
