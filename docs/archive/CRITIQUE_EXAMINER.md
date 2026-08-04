# CRITIQUE_EXAMINER.md: external-examiner critique + frontier programme (Fable, 2026-07-12)

> Requested by Kilian 2026-07-12: a holistic critique of the completed programme as an expert
> academic reviewer would grade it (fit against the five research objectives read verbatim from the
> assessed literature review; triviality; logical or argumentative errors in the storyline;
> methodological oversights), plus a brainstorm of future paths that would make the work more
> publishable and scientifically interesting, explicitly engaging Kilian's two named ideas
> (benchmarking against agentic LLMs; SBO for holistic contested supply-chain planning). Sources:
> the literature review PDF (§2.2 objectives verbatim), the full HANDOVER read order, every ledger
> gen01-gen23 plus a2/a3/a4/b4/d1/d2/d3/d3_gdansk/f3/zst_step0, the four prior critiques, and ONE
> new oracle-only probe run this session (`scratch/uniform_stack_probe.py`, numbers in §5.1; no
> training launched, no `src/` change). This file is a companion to `CRITIQUE.md`,
> `CRITIQUE_INTERDICTION.md`, `CRITIQUE_PREFREEZE.md` and `CRITIQUE_EXPANSION.md`; it deliberately
> does not repeat their findings except where the post-gen23 state changes the verdict.

---

## 0. Examiner's verdict in one paragraph

As an MSc research programme this is unusually complete, unusually honest, and clearly in the top
band on evidence: every objective has trained, pre-registered results; the negatives are mechanism
-backed rather than buried; the house discipline (pre-registration, oracle ground truths, fairness
rows, disclosed drift, dual selection) is itself examinable methodology. The two central positive
claims survive hostile reading: (1) on asymmetric interdiction instances an adversarially trained
SAC dispatcher approaches a computable minimax equilibrium (multi-convoy 0.256, 95% CI
[0.246, 0.266] vs equilibrium 0.206, n=10) where every deterministic planner and the
non-adversarial control are far more exploitable; (2) the map-conditioned generalist transfers
zero-shot to held-out cities (Gdansk 1.68x, Istanbul 1.88x its own equilibria) and the vanilla and
random-init controls bound it from both sides. The mark risks are no longer evidential. They are
(a) framing overclaims that a sharp examiner will find (ranked in §3), (b) two missing baselines
that the project's own logic demands (§4: an LP-distillation control for the ZST act; naive
-randomisation rows for the multi-convoy ladders, the latter measured this session and favourable),
and (c) compression and related-work positioning, where 50% of the rubric lives and where the March
survey contains no security-games section at all. Fix those and this defends as a distinction-band
thesis with two publishable spin-outs; leave them and the examiner writes the missing sentences for
you, less charitably.

---

## 1. Fit against the five research objectives (verbatim), post-gen23

The objectives from the assessed literature review §2.2, quoted verbatim, then the honest scoring.

**Obj 1. "Formulate the SDVRP as an asymmetric zero-sum Markov game, defining discrete action
spaces for a protagonist dispatcher and an environment-altering antagonist agent."**
Met, with deltas to declare. The formulation is realised most purely as a Stackelberg security game
with hidden pre-commitment (a strengthening over the promised simultaneous-move RARL game: say so).
gen20 closes the largest previous gap: a LEARNED antagonist now exists in a positive result
(strength 0.81x the oracle best response; the co-evolved defender lands at 0.330 +/- 0.018 vs the
oracle-trained 0.256 reference), with evaluation kept at oracle-BR portfolio-max so a weak learned
attacker cannot flatter the defender. Remaining honest sentence: in every HEADLINE ladder the
antagonist is still the oracle best response; the learned-adversary evidence is one co-evolution
cell, not the training regime of record. The "Markov game" wording also deserves one scoping
sentence: the headline games are repeated one-shot commitment games (state is degenerate); genuine
within-episode state enters only with gen19.

**Obj 2. "Design and implement a visual, interactive multi-agent simulation environment for the
SDVRP that serves as a training environment for training the SAC neural networks."**
Met, and exceeded on the environment side (five problem rungs, the interdiction layers, the
multi-city extraction pipeline with length repair, oracle fidelity gates, suite 161). The one
under-delivered word is "visual, interactive": the PyGame visualiser exists from the campaign era
but none of the interdiction-era exhibits are interactive. Cheap fix in §6, Tier 1.

**Obj 3. "Develop the SACRED framework, utilising a SAC architecture and ATLA, and to investigate
the efficacy of ERB bootstrapping via population-based metaheuristics to accelerate training
convergence and ensure policy coevolution."**
Met in an investigative sense, and this is fine IF written as such. ATLA is realised as smooth
fictitious play against a best response (declare the re-interpretation); SAC entropy is
load-bearing and positive. ERB bootstrapping now has a complete, literal answer with a mechanism:
gen23 seeded the buffer with ALNS-population demonstrations and found it HURTS (seeded plateaus at
0.443 and never reaches the 0.35 bar; cold reaches competence in 100-500 sorties, best-checkpoint
0.285), because deterministic metaheuristic demonstrations bias a mixed-strategy learner toward
exploitable determinism, while gen09's forced-copy from a MIXING leader helped. "We investigated
the efficacy and it is negative here, with a solution-concept-mismatch mechanism and a positive
contrast case" is a legitimate, arguably stronger, completion of the objective's verb than a bland
pass. Do not let any summary sentence say "ERB accelerates convergence": it did not.

**Obj 4. "Incorporate SBO into the SACRED framework, utilising a neural network metamodel to
approximate facility location and fleet composition, thereby enabling the holistic, simultaneous
evaluation of strategic supply chain design alongside the operations-level SDVRP."**
Met as a stack (F3 regression Spearman 0.894; D1 acquisition loop, median 32.5 evaluations to the
optimum vs random never; D2 hardening tier with the L1 = 0.29 equilibrium-shift interaction; D3
surrogate over the TRAINED policy, Spearman 0.959, policy-vs-oracle design-target correlation
0.768 in-distribution vs 0.109 on never-trained Gdansk). Two honest deltas: the demonstrated design
loops optimise the tiers SEPARATELY (the word "simultaneous" is not yet earned: §6 Tier 2 gives the
cheap experiment that earns it), and at K=1 sizes the exact evaluator is cheap, so the surrogate's
value is the loop pattern and the policy-valued target, never compute savings (already the recorded
position). One evidential wobble in the flagship exhibit is §4.4.

**Obj 5. "Evaluate the performance and resilience of the SACRED framework against SOTA adaptive,
population-based metaheuristics and a baseline non-adversarially SAC-trained agent under varied
levels of network disruption."**
Met, strongly: both headline ladders on corrected code with n=10 CIs (multi-convoy 0.256
[0.246, 0.266]; single-convoy paired dD 0.175 [0.137, 0.213] excluding zero, 10/10 seeds); the
disruption clause has 10/10 cells across K and N (gen12) plus zero-shot K/N shift rows; the
fairness rows (ALNS-forced-stack, vanilla best-checkpoint, fleet-cost premium equal to the
equilibrium's own) pre-empt the natural attacks; B4 makes independence a measured conservative
assumption. Two wording rules remain load-bearing: never call the in-house ALNS "SOTA" unqualified
(its defence is that it provably reaches loss_det, the optimum of the whole deterministic class);
and under a best-response metric ANY deterministic plan is maximally exploited, so the ladder needs
the naive-randomisation rows (§4.1, now measured) for "beats ALNS" to carry weight.

**The aim-level ZST promise.** Realised beyond the promise's plain reading: held-out ODs 1.59x,
held-out city 1.68x (select-on-train 1.73x), rotation to Istanbul 1.88x, whole-Kyiv 1.88x partial,
zero-shot K/N shift survives, and the causal control (gen21: a cost-trained generalist transfers at
2.34x, WORSE than random-init 1.99x) makes adversarial training the measured cause. The one control
the act still lacks is the amortisation baseline every ML examiner will ask for (§4.2).

---

## 2. Are the findings trivial?

Split the question, because parts of the result ARE guaranteed and the thesis wins by conceding
them crisply.

**Trivial (concede, one sentence each, never defend):**
- That a mixed strategy beats a deterministic route against a best-responding interdictor is
  minimax arithmetic from 1953, and the equilibrium ladder positions of shortest-path/ALNS are
  computed, not discovered.
- That SACRED "beats ALNS" is close to structural: ALNS emits a deterministic plan, which the
  metric maximally punishes; on 35-159 even non-adversarial vanilla (0.526) beats ALNS (0.699).
  The load-bearing comparisons are vs the equilibrium (1.24-1.33x), vs vanilla, and vs the naive
  -randomisation rows (§5.1).
- F3/D1 in isolation: a surrogate fed a feature that is the closed-form equilibrium predicting the
  equilibrium, on an enumerable design space, is demonstration, not discovery.

**Not trivial (the actual contributions, in my ranking):**
1. **The learning-dynamics account**: WHEN does deep RL find the security-game equilibrium?
   Instance asymmetry decides learnability (the 72-pair disjoint screen; the 62-97 vs 35-159
   contrast); the fictitious-play discipline bracket (pure-BR cycles, stale mixtures park,
   smooth-FP passes, and the smoothing that fixes shared-edge instances collapses disjoint ones);
   the equilibrium as a reproducible TRANSIENT (four failed hold-the-tail attempts) with
   best-checkpoint selection as the honest resolution; identity-vs-semantics representation
   effects (gen11b). No LP produces any of this, and it is the part an RL venue would referee as
   novel.
2. **The ZST arc with its causal control**: one map-conditioned policy at 1.6-1.9x equilibrium on
   never-seen cities, bounded by random-init above and equilibrium below, with the vanilla control
   showing cost training actively destroys transfer. The measured transfer-difficulty ladder
   (OD -> city -> construction pipeline) is the right shape of claim.
3. **The mechanism-backed negatives**: gen23 (demonstrations of the wrong solution concept hurt),
   gen18 (coordination fails exploration-side even with trained credit signals), gen17 (transience
   is FP-inherent), the campaign (flat attack landscapes defeat adversarial training, with
   preconditions). These are the parts that make the thesis trustworthy.
4. **The objective-is-load-bearing finding**: risk-neutral objectives let deterministic spreading
   substitute for randomisation; loss-aversion is where unpredictability pays. Currently a binary
   contrast; §6 Tier 2 turns it into a law.
5. **D3's policy-valued design target** (0.768 vs 0.109): the claim that strategic design should
   price the DEPLOYED policy, not the equilibrium abstraction, is genuinely novel in this setting,
   pending the §4.4 reliability check.

**Prior-art caution for (2) and (5):** the thesis positions itself against deployed security games
(ARMOR/IRIS lineage) and network interdiction (Wood; Washburn and Wood), but the ML-adjacent
literatures of amortised game solving and learning-to-optimise (supervised nets that map instance
-> equilibrium), DeepFP-style differentiable game solvers, and GNN generalisation for combinatorial
optimisation are closer neighbours of the ZST act than anything currently cited. The novelty
sentence that survives review is roughly: "a model-free adversarially trained policy that
approximates minimax play on real road networks, transfers zero-shot across cities, and composes
with design optimisation, WITHOUT equilibrium labels": which is defensible, but only alongside the
distillation control of §4.2.

---

## 3. Logical and argumentative errors in the storyline (ranked by examiner danger)

3.1 **"All five objectives met" shorthand.** The banners use it; the conclusions chapter must not.
The honest per-objective deltas (§1) cost half a page and buy immunity. (Already planned as
NEXT_STEPS item 7; keep it non-negotiable.)

3.2 **The Act-III pivot anecdote is over-read.** THESIS_STORYLINE presents gen05's "+1667 against
the competent deterministic victim" as the campaign's own data containing the seed of the
exploitability reframe. gen07's corrected gate later showed that BR was a transfer artefact (a BR
trained against exploitable learned policies, applied to greedy), not evidence of learnable attack
structure. Keep the anecdote as MOTIVATION ("what made us look"), never as EVIDENCE; the evidence
for the pivot is the flat-landscape mechanism plus the interdiction equilibrium probe.

3.3 **"Reactivity is useless against interdiction" is broader than the evidence.** It is true
within a sortie against a pre-committed hidden ambush (structural), but gen19 itself shows
adaptation across sorties is worth 3x (0.147 -> 0.050) against a pattern-of-life adversary, i.e.
the thesis's own D-act restores a form of reactivity. Scope the sentence to "en-route rerouting
cannot undo exposure to a pre-committed ambush" (untested, flagged as such, or tested cheaply via
B1-lite-2) and let anticipation-vs-reaction be the nuanced story the Ritzinger framing deserves.

3.4 **gen06's "definitive, significantly reversed" leans on the pooled analysis.** dD_targeted =
-881 +/- 284 treats 90 paired instances across 3 seeds as independent; the ledger's own
conservative seed-level test (n=3, CI [-2017.7, +255.3]) does not exclude zero. The finding is
real as a mechanism chain replicated across generations; the statistical wording in the thesis
should be "consistent and mechanism-backed across arenas, though under-powered at seed level per
generation", not "significant". The campaign's persuasive unit is the chain, not any single p.

3.5 **Verdict inflation in three expansion ledgers.** (a) gen22 is headlined PASS while its
pre-registered ">= 4/6 loss_det" clause held on 1/3 seeds only (disclosed in the body; the header
should say PASS-with-a-miss). (b) gen21's "adversarial training is causal for ZST" rests on one
vanilla seed; the direction is clean but the wording should be "a measured control, n=1" until
§6 Tier 1 triples it. (c) gen20's bar (within 0.10 of the oracle-trained reference) is generous
and seed 0 sits on it (0.355 vs 0.356); report the margin, and frame the "campaign reversal"
("the learned adversary that could not learn congestion CAN learn interdiction") as a
cross-game consistency, not a controlled comparison: the two games differ in far more than
attack-surface sharpness.

3.6 **The best-checkpoint story has an unconfronted pattern: best is EARLY.** Across gen13/15/16/
20/22 the selected checkpoint sits at sorties 200-500 of 1200+ while final iterates drift to
0.56-2.6x worse. Selection is disclosed, exact (the evaluator is not noisy, so the min is not a
statistical fishing trip on the training instance), and dual-reported for ZST; but the thesis
nowhere says plainly that under FP dynamics most of the training run is spent LEAVING the
equilibrium, and that the deployable artefact is an early-stopped policy whose stopping signal
needs the oracle (or train-instance exploitability) to compute. One honest paragraph converts
this from a discovered weakness into a reported property of last-iterate FP, and connects it to
the known non-convergence of last-iterate fictitious play (cite the learning-in-games
literature; average-iterate converges, last iterate need not). Related future-work sentence:
optimistic/extragradient dynamics are the algorithm family with last-iterate guarantees and were
deliberately not chased (gen17 closed the FP-family question); that is the correct scoping of
"inherent".

3.7 **gen19's register must never blur into the minimax register.** The pattern-of-life adversary
is a quantal-response, bounded-memory behavioural model; iid_eq (0.147) sits BELOW V_eq (0.206),
and SACRED's 0.050 ~ history_opt 0.049 is exact-DP-matching in a stationary MDP, not equilibrium
play. The ledger largely handles this (worst-case row 0.219 vs 0.206; QR framing), and the RVI
-solvable structure is a strength (a computable dynamic yardstick); the storyline sentence just
needs to say "exploits a boundedly rational adaptive adversary at no meaningful worst-case
premium", never "approaches the equilibrium".

3.8 **The headline-instance migration (62-97 -> 35-159) needs one prevalence sentence.** The move
was forced by an honest bug fix and the screens were pre-registered, but the pattern "the
representation fix regressed the flagship instance, and the flagship then moved to an instance
where the plain pipeline works" is exactly what a forking-paths reviewer probes. Two things
defuse it: the disclosed mechanism (gen11b: the pre-fix number was partly route-identity
memorisation; honest embeddings need instance asymmetry), and a POPULATION statement: what
fraction of high-connectivity OD pairs have enough asymmetry for calibrated mixing to matter?
The 72-pair screen and the 20-pair mission-gap scan partially answer this; §6 Tier 1 completes
it for pennies.

3.9 **d3_gdansk's 0.109 is currently ambiguous between signal and noise** (see §4.4). Do not put
it on the poster until the reliability check runs.

3.10 **The SDVRP title tension stands.** The S lives in Acts I-II only; the D returns in lite form
(gen19); the headline game has no demand, no capacities, no multi-destination routing: it is the
route-choice kernel of contested resupply, not a VRP. The recorded one-sentence concession (a
deliberate refinement, evidenced by the negative campaign) is the right move; §6 Tier 3 sketches
the multi-OD game that would materially close the gap, and full B1 remains the recorded
extension.

---

## 4. Methodological oversights (new ones; the four prior critiques' lists are not repeated)

4.1 **The multi-convoy ladders lack any naive-randomisation row: MEASURED AND CLOSED THIS SESSION.**
The single-convoy ladder always carried uniform (0.455); the multi-convoy ladders never had an
equivalent, leaving the sharpest cheap attack open: "does the deep-RL apparatus beat the 3-line
heuristic 'stack the fleet on one uniformly-random route'?" The new oracle probe
(`scratch/uniform_stack_probe.py`) answers it, favourably (§5.1): uniform-stack scores 0.442 on
35-159 vs SACRED 0.256, so calibrated randomisation carries a 42% relative margin over the trivial
heuristic. Fold both rows (uniform-stack, uniform-independent) into the gen13/gen14 ladders and the
thesis figure; the claim gets stronger and the hole closes.

4.2 **The ZST act lacks the amortisation control: LP-distillation.** gen21 rules out cost-training;
it does not rule out the standard ML alternative an examiner will name: supervised amortisation of
the solver (train the SAME architecture to imitate the oracle equilibrium mixture on the training
instances, no adversary, then evaluate zero-shot on Gdansk/Istanbul under the same oracle-BR ratio
metric). Labels are free at K=1 (the LP is milliseconds). Every outcome is citable: if SACRED beats
distillation, adversarial interaction contributes beyond label-fitting; if they tie, adversarial RL
is an equilibrium amortiser that needs no labels, and the honest claim becomes exactly that, plus
the boundary fact that labels stop existing past the enumeration wall (K >= 4, A4's regime) where
only self-play can train; if distillation wins, the ZST act is re-scoped and the thesis is saved
from overclaiming before an examiner does it. This is the highest-value ~1-2 days left on the
board and I would run it before any other new experiment.

4.3 **gen21 is n=1.** Two more vanilla-generalist seeds (~2-3 h total) upgrade the causal sentence
from "a measured control" to a defensible comparison. Cheap; do it whenever a machine is idle.

4.4 **d3_gdansk's headline correlation (0.109) has no reliability denominator.** The 0.768 -> 0.109
collapse is read as "on an unseen theatre you must design against the deployed policy", but the
same number is produced if the policy-exploitability TARGET on Gdansk designs is simply noisy
(test-retest unreliability) even though gen16 shows the policy itself is decent there (1.68x).
Eval-only fix, hours: re-evaluate the design sweep with fresh eval seeds, report the test-retest
correlation of the target, and present the disattenuated policy-vs-oracle correlation. If the
target is reliable and the correlation still collapses, the poster claim is earned and stronger;
if not, the exhibit moves to "in-distribution only" honestly.

4.5 **The route-menu-relative equilibrium: MEASURED AND CLOSED THIS SESSION.** The equilibrium the
whole programme scores against is defined over the k-shortest candidate menu, which was a fair
"both sides see the same game" convention but had never been sensitivity-checked. The probe shows
the game value saturates by k_extra=4 (0.2061, stable through R=20) on 35-159, so the k8 menu is
sufficient and one sentence plus a citation closes the scope question. Bonus finding worth a
figure: uniform-stack DEGRADES with menu size (0.25 at R=4 -> 0.53 at R=20) while the equilibrium
is flat, i.e. the value of calibration GROWS with the route menu: naive randomisation does not
scale, calibrated randomisation does.

4.6 **The interception model is calibrated by construction, not by data.** Vulnerabilities are a
length-derived band; nothing validates it against any real interdiction geometry, and the thesis
should say once that all claims are conditional on the threat-map model class, with the D2/B4
sweeps as the sensitivity story. (This is a scope sentence, not an experiment.)

4.7 **Statistical hygiene at the expansion tail.** gen15/16/17/18/20/22 are n=3 with population
std; gen21 n=1; several eval-only rows are single-checkpoint. The gen14 template (n=10, t-CIs,
per-seed lists) should be stated as the evidentiary standard, with the n=3 tables captioned as
such (already the recorded rule: enforce it in the writing).

---

## 5. New measurements produced by this critique (oracle-only, no training)

### 5.1 The missing naive-randomisation rows (probe: `scratch/uniform_stack_probe.py`)

Mission-failure exploitability under the oracle best response, N=3, K=1, band (0.15, 0.95),
absolute normalisation, k_extra=8:

| arm | 35-159 (headline) | 62-97 (pre-fix) |
|---|---|---|
| loss_det (= ALNS optimum) | 0.699 | 0.699 |
| uniform-INDEPENDENT (each convoy uniform) | 0.546 | 0.848 |
| uniform-STACK (all on one uniform route) | 0.442 | 0.649 |
| equilibrium (loss_mixed) | 0.206 | 0.216 |

Read: SACRED's banked 0.256 [0.246, 0.266] beats the strongest naive-randomisation heuristic by
0.186 on the headline instance (and the pre-fix 0.295 beat 0.649 on 62-97). The rows belong in
every multi-convoy ladder; they convert "beats ALNS" (structurally cheap) into "beats every
uncalibrated strategy class: deterministic, independent-mixing, and stack-and-randomise-uniform"
(the real claim).

### 5.2 Menu sufficiency (same probe, 35-159)

| k_extra | R | equilibrium | loss_det | uniform-stack |
|---|---|---|---|---|
| 0 | 4 | 0.2411 | 0.7187 | 0.2500 |
| 4 | 8 | 0.2061 | 0.6994 | 0.3471 |
| 8 | 12 | 0.2061 | 0.6994 | 0.4421 |
| 12 | 16 | 0.2061 | 0.6994 | 0.4974 |
| 16 | 20 | 0.2061 | 0.6994 | 0.5305 |

Read: the equilibrium is menu-stable from R=8, so the k8 yardstick is not an artefact of menu
truncation; and the widening uniform-stack column is the "value of calibration grows with the
route set" exhibit described in §4.5.

---

## 6. Future paths (the requested brainstorm), ranked in three tiers

Calendar rails: FAR + presentation 30 July; experimental freeze 3 August (HARD); thesis + poster
10:00, 28 August, 12,000 words. Today is 12 July: about three experimental weeks IF writing starts
inside them. Every launch remains Kilian's explicit go; each item gets its own pre-registered
ledger.

### Tier 1: close the sharpest attacks (this week; days, mostly eval-only)

1. **Fold the §5 rows into the ledgers and figures** (an hour). Ladder rows for gen13/gen14; the
   menu-sufficiency sentence; the calibration-vs-menu-size mini-figure.
2. **The LP-distillation generalist control** (§4.2; ~1-2 days, the one new TRAINING run I would
   fund first). Same encoder and menu head, KL loss to the per-instance oracle equilibrium
   occupancy over the gen16 training pool, zero-shot evaluation on Gdansk and Istanbul under the
   identical ratio metric, dual-selection reporting. Decision-grade whatever the outcome.
3. **gen21 to n=3** (§4.3; ~2-3 h). Upgrades the causal-transfer sentence.
4. **d3_gdansk reliability check** (§4.4; hours, eval-only). Test-retest the policy-exploitability
   target; report the disattenuated correlation; poster claim gated on it.
5. **The prevalence figure** (§3.8; half a day, oracle-only, free). Over every high-connectivity
   OD pair in all four cities: compute uniform-stack/equilibrium and loss_det/equilibrium ratios;
   plot the distribution; mark the headline instances on it. One figure answers "how often does
   calibrated mixing matter, and did you cherry-pick?" for the whole thesis.
6. **One interactive exhibit** (half a day). The Obj-2 "visual, interactive" wording plus the
   poster both want it: an HTML map of a held-out city where the reader toggles
   shortest-path/ALNS/uniform-stack/SACRED and sees route distributions plus interception
   numbers. All numbers exist; this is presentation, and 20% of the rubric is presentation.

### Tier 2: the differentiators (next 1-1.5 weeks; each is one bounded ledger)

7. **Agentic-LLM exploitability benchmark** (Kilian's idea; ~1-2 days, eval-only, no training).
   The literature review's own title promises "Agentic AI", and no adjacent work scores LLM
   planners on a security-game yardstick. Design: give 2-3 frontier LLMs the full instance
   specification (routes, vulnerabilities, K, objective) in three registers: (a) "choose a route"
   (deterministic register), (b) "output a probability distribution over the route menu"
   (strategy register: score the STATED mixture exactly under the oracle BR, the same TAP-style
   arithmetic as every other arm), (c) sequential play over T sorties with interception feedback
   (agentic register, scored on realised occupancy; optionally against the gen19 pattern-of-life
   adversary). Pre-register the ladder positions as hypotheses (register (a) lands at loss_det;
   (b) lands between uniform and equilibrium, miscalibrated; (c) tests whether in-context
   adaptation moves it). Pin model versions, temperature, prompts; log transcripts as the
   reproducibility record. Risks: an examiner reading it as a gimmick (mitigated by identical
   pre-registration discipline and by placing it as one Results subsection plus one ladder
   column); LLM randomisation quirks are themselves a known literature (cite it). Payoff: a
   headline table nobody else has, a self-contained workshop-paper spin-out, and a second
   independent justification for the thesis's central mechanism (calibrated randomisation is
   exactly what language agents cannot do unaided).
8. **The holistic-SBO integration-gap experiment** (Kilian's idea; ~1-2 days on existing
   machinery). The D-chain optimises tiers separately; Obj-4's verbatim promise is "holistic,
   simultaneous". Design: joint space = placement x hardening allocation x fleet size; arm A
   optimises sequentially tier-by-tier (the classical decomposition, D1 then D2 then N), arm B
   runs ONE surrogate-guided loop over the joint space at the SAME total evaluation budget; both
   evaluated by the frozen generalist's exploitability (plus cost) on the composed design;
   pre-registered primary = the integration gap (joint minus sequential) with the D2 interaction
   (L1 = 0.29) as the mechanism story. Run it on Gdansk with the multi-city generalist and it is
   simultaneously the strongest Obj-4 sentence and the poster exhibit ("design a supply chain in
   a never-seen theatre in one loop; no exact method can price it"). If the gap is ~0, that too
   is a finding (the tiers decompose; say why).
9. **The risk-aversion spectrum** (half a day oracle-only; optionally +1 trained cell). The
   objective finding is currently binary (risk-neutral: gap collapses; mission: gap holds). The
   oracle already supports threshold objectives (m of N), so sweep the loss-aversion parameter
   (and/or a CVaR-alpha weighting) and plot equilibrium-vs-deterministic gap against it: "the
   price of predictability as a function of loss-aversion", one curve that generalises the
   modelling choice into a law and immunises the mission objective against the "you picked the
   objective where you win" reading (the answer becomes: here is exactly where winning starts).
10. **Oracle-level probe for the multi-OD game** (free; the gateway to Tier 3 item 12). Extend the
    oracle to convoys with DIFFERENT destinations sharing corridor edges (joint route tuples:
    R1 x R2 x R3 is LP-tractable at menu sizes) and measure the CORRELATION GAP: best correlated
    joint mixture vs best product-of-independents. If the gap is material, the richer game exists
    and item 12 is justified; if not, that is a clean scoping result for free.

### Tier 3: bigger swings (post-freeze or publication-phase; do not start before Tier 1-2 land)

11. **PSRO / double-oracle population** on a headline instance (2-3 days): upgrades gen20 from one
    learned attacker to a principled population loop, connects to Obj-3's "population-based"
    wording a second way, and is the standard scalable-solver story for large K (composes with
    A4's greedy BR). The publication-grade version of the adversary axis.
12. **The multi-OD interdiction game** (3-5 days if item 10's probe says yes): one base, several
    FOBs, shared corridors. Kills the "all convoys share one OD" objection, makes the coordination
    content genuinely joint (correlation without stacking), and moves the game materially toward
    the VRP the title promises. New learnability risk is real (the gen18 boundary); gate on the
    oracle probe and a smoke.
13. **A4's K=5 trained cell** (deferred 2026-07-11; keep deferred unless a free day appears): the
    one datapoint that makes "trained where exact solvers cannot follow" a measurement, reported
    with the (1 - 1/e) certified interval and the column-generation concession.
14. **B1-lite-2 (en-route threat revelation)** (~2-3 days): tests §3.3's scoped reactivity
    sentence directly and restores the within-sortie D; backward-induction yardstick already
    specced.
15. **Optimistic/extragradient last-iterate attempt**: the only algorithm family with last-iterate
    guarantees, deliberately out of scope for the thesis (gen17's gate stays closed); first
    experiment of any follow-on paper on the dynamics thread.
16. **Release the environment + oracles as a benchmark** ("contested-logistics gym": envs, LP/
    greedy oracles, city pipeline, ladders as reference solutions; a day of packaging post
    -submission). Cheap, citable, and the artefact reviewers of any spin-out paper will ask for.
17. **Full B1 (Poisson demand campaign) and B5 (deception/decoys)**: remain the recorded
    extensions; neither fits before the freeze; both are paper-scale ideas, and B5 is where the
    security-game theory gets deepest (belief manipulation).

### Publication map (where the spin-outs would go)

- **Security-games/agents venue (AAMAS, IJCAI multi-agent track):** the interdiction results +
  ZST + learned-adversary co-evolution, positioned against deployed security games and network
  interdiction; needs the distillation control and the prevalence figure.
- **RL venue or journal (RLC/TMLR):** "when adversarial RL helps: flat vs peaked attack
  landscapes", the preconditions, the FP-dynamics bracket, the transient finding: the negative
  campaign plus dynamics act is a standalone methods paper.
- **OR journal (EJOR / Computers & OR):** the holistic-SBO integration story (items 8-9 + D-chain)
  in the contested-logistics application.
- **Workshop (NeurIPS/ICLR agents or eval track):** the LLM exploitability benchmark (item 7),
  fast to write once measured.

---

## 7. Firm recommendation (what I would actually do, in order)

Week of 14 July: Tier 1 complete (items 1-6; one training run among them: the distillation
control), plus gen21 seeds. Week of 21 July: items 7 and 8 (the two Kilian-named differentiators;
both bounded, both eval-heavy), item 9's oracle sweep, item 10's probe; draft the FAR from the
chronicle in parallel and hold the 30 July deadline sacred. Then NEXT_STEPS item 7 (the storyline
rewrite absorbing §1-§4 of this file) and writing to the freeze, with Tier 3 untouched before
3 August unless something above finishes early. The single most important experiment left in the
project is item 2 (LP-distillation): it is the one result that changes what the central ZST claim
is allowed to say, and every week it stays unrun is a week the thesis risks being written around a
sentence an examiner can delete.

*Artefacts of this critique: this file; `scratch/uniform_stack_probe.py` plus the §5 numbers
(oracle-only, reproducible in seconds). No training launched; no `src/` changes; suite untouched.*
