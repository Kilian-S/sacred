# CRITIQUE_12-07-26.md: independent holistic critique + future-path programme (Fable, 2026-07-12)

> Requested by Kilian 2026-07-12: a fresh, holistic critique of the whole SACRED interdiction
> programme as an expert academic reviewer would grade it (fit against the five research objectives
> read verbatim from the assessed literature review; triviality; logical and argumentative errors in
> the storyline; oversights in approach and methodology), plus a ranked brainstorm of future avenues
> that would make the project more publishable and scientifically interesting, explicitly engaging
> Kilian's two named directions (benchmarking against agentic LLMs; SBO for holistic contested
> supply-chain planning) without being limited to them.
>
> **Sources:** the complete HANDOVER read order (HANDOVER, REDESIGN_INTERDICTION, THESIS_STORYLINE,
> SACRED_PROGRESS entries 1-21, ROADMAP, DIRECTION, DIRECTION_EXPANSION, CONTEXT, PROBLEM_REDESIGN,
> SYSTEM, TASK, NIGHT_REPORT, NEXT_STEPS); every ledger gen01-gen23 plus
> a2/a3/a4/b4/d1/d2/d3/d3_gdansk/f3/zst_step0; all four prior critiques (CRITIQUE 2026-07-02,
> CRITIQUE_INTERDICTION 2026-07-09, CRITIQUE_PREFREEZE 2026-07-10, CRITIQUE_EXPANSION 2026-07-11)
> and this morning's CRITIQUE_EXAMINER (2026-07-12); the literature survey PDF (§2.1 aim + §2.2
> objectives, read verbatim) and the guidance PDF (rubric: Methodology/Analysis/Discussion 50%,
> Structure & Presentation 20%, Abstract/Intro/Conclusions 10% each); the load-bearing code
> (`length_band_vulnerability`, the featurisation path, the generalist trainer, the oracles); and
> **one new oracle-only probe run this session** (`scratch/threatmap_geometry_probe.py`, numbers in
> §3.1; seconds of CPU, no training, no `src/` change).
>
> **Positioning:** this file deliberately does NOT repeat the findings of the five prior critiques.
> Where I re-examined a prior verdict and concur, I say so in one line and cite it. The core of this
> file is §3 (findings no prior critique caught), §4-5 (new argumentative and methodological gaps),
> and §6-7 (the future-path programme). CRITIQUE_EXAMINER.md (this morning) remains the reference
> for the ranked storyline-error list and the naive-randomisation/menu-sufficiency probes; this file
> is its adversarial second opinion.

---

## 0. Verdict in one paragraph

I read the entire record adversarially, looking for the claim an examiner could delete, and I
concur with the standing assessment that the evidence base is unusually complete and unusually
honest for an MSc: two headline ladders on corrected code with n=10 confidence intervals, a
zero-shot transfer arc bounded by controls on both sides, a full SBO stack, a learned-adversary
positive, and mechanism-backed negatives that strengthen rather than embarrass the story. The
discipline (pre-registration, pinned SHAs, oracle ground truths, disclosed drift, fairness rows,
two survived retractions) is itself distinction-band material. My independent pass found **one
structural weakness no prior critique has named**: every threat map any policy has ever trained on
or been evaluated against is, by construction, an affine transform of edge length
(`length_band_vulnerability`), so the flagship "map-conditioned" transfer claim is currently
observationally equivalent to "geometry-conditioned", and the decisive experiment separating the
two (a shuffled-map transfer row, eval-only) has never run (§3.1). Second, the ratio-to-equilibrium
metric that carries the transfer ladder systematically flatters cells with thin
deterministic-to-equilibrium headroom: restated as the fraction of the gap actually closed, the
transfer ladder decays from roughly 90% (trained instance) through ~52% (held-out city) to ~24%
(Istanbul) and ~8% (whole Kyiv), which is a materially more honest and, presented well, a more
interesting shape than "1.9x its equilibrium" (§3.2). Both are fixable cheaply before the freeze,
and both change wording rather than results. With those fixed, plus the LP-distillation control the
examiner critique already demands, the thesis defends at the top of the band and carries two to
three publishable spin-outs.

---

## 1. Fit against the five research objectives (verbatim), second opinion

The objectives from the assessed literature review §2.2, quoted verbatim, with my scoring. Where I
concur with CRITIQUE_EXAMINER §1 I compress to a line; deltas that are new are marked **NEW**.

**Obj 1. "Formulate the SDVRP as an asymmetric zero-sum Markov game, defining discrete action
spaces for a protagonist dispatcher and an environment-altering antagonist agent."**
Met with declared deltas (concur: Stackelberg security game rather than simultaneous-move RARL; the
headline antagonist is an oracle best response with gen20 supplying the learned-agent positive; the
headline games are repeated commitment games, with genuine within-episode state only in gen19).
**NEW:** the thesis should add one game-theoretic sentence that buys real rigour: in zero-sum
games the Stackelberg (commit-to-a-mixed-strategy) value and the maximin value coincide, which is
exactly why the "hidden pre-commitment" story and the minimax LP yardstick are the same number
(the interchangeability result of Korzhyk, Yin, Conitzer et al.; verify the reference before
citing). Without that sentence, a game-theory examiner may probe why a Stackelberg framing is
scored against a Nash/minimax LP; with it, the framing is airtight and the pattern-of-life
justification (the attacker observes repeated play, hence best-responds to the mixture) connects
cleanly to gen19.

**Obj 2. "Design and implement a visual, interactive multi-agent simulation environment for the
SDVRP that serves as a training environment for training the SAC neural networks."**
Met and exceeded environment-side; the "visual, interactive" wording is under-delivered for the
interdiction era (concur; the examiner critique's HTML-exhibit fix is the right one and doubles as
poster material).

**Obj 3. "Develop the SACRED framework, utilising a SAC architecture and ATLA, and to investigate
the efficacy of ERB bootstrapping via population-based metaheuristics to accelerate training
convergence and ensure policy coevolution."**
Met in the investigative sense, and gen23's negative-with-mechanism (deterministic metaheuristic
demonstrations bias a mixed-strategy learner toward exploitable determinism; the mixing-leader
demonstrations of gen09 are the positive contrast) is a stronger completion of "investigate the
efficacy" than a bland pass. Concur with the standing rule: no summary sentence may say ERB
accelerated anything. **NEW:** the gen23 mechanism deserves one precise formulation in the thesis,
because it generalises: *demonstration data helps exactly when the demonstrated behaviour lies in
the support of the target solution concept; a deterministic expert has measure zero in a
mixed-strategy equilibrium.* That sentence converts a null into a citable principle and connects
gen23, gen09-M4 and the imitation-learning literature.

**Obj 4. "Incorporate SBO into the SACRED framework, utilising a neural network metamodel to
approximate facility location and fleet composition, thereby enabling the holistic, simultaneous
evaluation of strategic supply chain design alongside the operations-level SDVRP."**
Met as a stack (F3 -> D1 -> D2 -> D3 -> D3-on-Gdansk); "simultaneous" is not yet earned because the
tiers are optimised separately (concur; the integration-gap experiment in §6 earns it). **NEW:**
the design loops are single-objective (exploitability) while the operational act explicitly prices
the security premium in fleet cost (123.1 vs ALNS 96.1). A strategic designer would face the same
trade-off, so the design act should show at least one bi-objective (cost, exploitability) frontier
over designs. Cheap: both quantities already exist per design; it is one scatter plot and it
pre-empts "you designed for security at any price".

**Obj 5. "Evaluate the performance and resilience of the SACRED framework against SOTA adaptive,
population-based metaheuristics and a baseline non-adversarially SAC-trained agent under varied
levels of network disruption."**
Met, strongly (concur on every clause: n=10 CIs, 10/10 disruption cells, fairness rows, the
naive-randomisation rows now measured, the ALNS-reaches-loss_det certificate framing instead of
"SOTA"). **NEW, small:** the objective says metaheuristic**s**, plural. The certificate argument
covers the plural (any deterministic planner, however sophisticated, is bounded by loss_det, which
ALNS provably attains), but the thesis should make that plural-coverage argument explicitly rather
than leave "we implemented one 130-line ALNS" as the surface reading.

**The aim-level ZST promise ("resilient, zero-shot transferable logistics policies that standard
algorithms cannot achieve").**
Realised at OD, city, rotation and scale levels with a causal control (gen21) and bounded above by
random-init. Two new caveats, both material, in §3.1 and §3.2: the mechanism wording
("conditioned on the threat map") is currently unproven against the geometry-conditioning
alternative, and the ratio metric overstates how much calibration content actually transfers at
the far end of the ladder.

---

## 2. Are the findings trivial?

I concur with CRITIQUE_EXAMINER §2's split (concede the minimax arithmetic, the structural
"beats-ALNS", and the F3-in-isolation demonstration; the real contributions are the
learning-dynamics account, the ZST arc with controls, the mechanism-backed negatives, the
objective-is-load-bearing finding, and the D3 policy-valued design target). I add one sharpening
that no prior critique states plainly:

**The deployable-artefact question.** On a fixed instance, the deployable object the whole
single-instance apparatus produces is a route-frequency table (the best-checkpoint TAP mixture).
The LP emits the equilibrium route-frequency table for the same instance in milliseconds. So for
any FIXED instance, the trained network is not the artefact; the *scientific product* of the
single-instance acts is the learning-dynamics account (when and how model-free adversarial
learning finds, overshoots and leaves the equilibrium), and the *artefact* claim of the thesis
lives entirely in the generalist (one network, many instances, zero-shot) and in D3 (a design loop
priced by the deployed policy). The thesis should say this in one sentence in the discussion,
because an examiner who formulates it first will use it as a wedge; stated proactively, it is
simply the correct division of labour between the acts, and it makes the ZST act the load-bearing
one, which the evidence supports.

---

## 3. New findings of this critique (not in any prior critique)

### 3.1 Every threat map in the programme is a deterministic transform of geometry; the "map-conditioned" transfer claim is untested against the "geometry-conditioned" alternative

**The fact (from code, not from the ledgers):** every instance in every act builds its
vulnerability map with `length_band_vulnerability` (`src/baselines/interdiction_oracle.py:180`):
each candidate edge's length is mapped affinely into the band (0.15, 0.95) (absolute normalisation
across the graph since gen09). The "observable threat map" that the generalist receives as an edge
column, and the per-route [cost, worst-vulnerability] head features, are therefore functions of
edge lengths only. No trained policy in the project's history has ever seen a threat map that
carries information beyond the road geometry it can already observe.

**Measured consequence (new probe, `scratch/threatmap_geometry_probe.py`, oracle-only, seconds):**
on 8/8 instances sampled by the gen15 pool recipe (Kaliningrad, N=3, K=1, band 0.15-0.95, k8):

| OD | corr(route cost, route worst-vuln) | eq (true map) | eq (edge-permuted map, 5 shuffles) | leader-mass L1 shift |
|---|---|---|---|---|
| 72-42 | +0.867 | 0.326 | 0.273 +/- 0.038 | 0.60 |
| 103-27 | +0.862 | 0.279 | 0.275 +/- 0.010 | 0.89 |
| 66-230 | -0.615 | 0.326 | 0.257 +/- 0.026 | 0.88 |
| 219-19 | +0.600 | 0.277 | 0.298 +/- 0.007 | 0.75 |
| 64-11 | -0.855 | 0.326 | 0.273 +/- 0.023 | 0.44 |
| 189-11 | +0.995 | 0.327 | 0.310 +/- 0.017 | 0.71 |
| 26-158 | +0.843 | 0.298 | 0.321 +/- 0.010 | 0.80 |
| 13-31 | -0.887 | 0.318 | 0.275 +/- 0.018 | 1.03 |

Three readings. (a) The two features the generalist head reads are strongly collinear (|corr|
0.60-0.99 in 8/8 instances; the sign varies because worst-vuln tracks the longest edge while cost
tracks the sum, but both derive from the same lengths). (b) When the same vulnerability values are
randomly permuted across edges (a map genuinely decorrelated from geometry), the equilibrium
VALUE moves only modestly, but the equilibrium STRATEGY moves a lot (leader-mass L1 shifts of
0.44-1.03 on a simplex where the maximum is 2.0). So *where* the hedge should sit depends
strongly on *where* the map puts danger, and a policy that truly reads the map must track those
shifts. (c) Nothing in the current evidence can distinguish a policy that reads the threat map
from one that reads road geometry, because the two have never been decoupled.

**Why it matters:** the ZST act's mechanism sentence ("give the policy the map and transfer
works", gen15; "the map-conditioning is the invariant", gen16 K/N rows) is the thesis's central
positive mechanism claim, and, as worded, it is unproven. It also touches the operational story:
the interesting deployment case is precisely intel-driven threat maps (watched chokepoints,
ambush intelligence) that do NOT follow geometry; a defender who only ever learned
"long edges are dangerous" has learned a static property of the world, not map-conditioning. Note
the project has already met this issue once from the other side: the I3 wave-1 failure was caused
by vulnerability correlating with cost (vanilla imitated the equilibrium), and the descending-band
variant was probed and killed at the oracle level. The lesson was applied to instance design for
the sacred-vs-vanilla comparison, but never to the transfer act.

**The decisive experiment (eval-only, half a day, decision-grade either way):** take the frozen
gen16 multi-city generalist; on the held-out Gdansk ODs, permute each instance's vulnerability
values across its candidate edges (as in the probe); recompute each shuffled instance's oracle
equilibrium and BR; feed the policy the SHUFFLED map (the observation column and route features
recompute mechanically); report the ratio-to-equilibrium on shuffled maps beside the gen16 1.68.
If the policy tracks the shuffled equilibria (ratio comparable to 1.68, beats random-init), the
map-conditioning claim is genuinely earned and materially STRONGER than anything currently
claimed (it would show transfer across threat fields, not just cities). If it collapses toward
random-init, the honest wording becomes "conditioned on road geometry under a
geometry-consistent threat model", and the fix (a training pool with randomised maps; the
machinery exists, roughly one gen16-scale retrain) is the obvious pre-freeze candidate. I would
rank this experiment equal-first with the LP-distillation control: both decide what the flagship
claim is allowed to say.

### 3.2 The ratio-to-equilibrium metric flatters thin-headroom cells; restate the transfer ladder as gap closure

Per-instance performance is reported as ratio = policy exploitability / equilibrium. Ratios are
comparable across instances only if the deterministic-to-equilibrium headroom is comparable, and
at the far end of the transfer ladder it is not (the ledgers themselves note "thin-asymmetry ODs"
in Kyiv and Istanbul). The metric that measures what the thesis actually claims (calibrated
randomisation beyond what any deterministic planner achieves) is the **gap-closure fraction**:
(loss_det - policy) / (loss_det - equilibrium), i.e. how much of the deterministic-to-equilibrium
gap the policy closes; 1.0 = equilibrium play, 0 = no better than the deterministic optimum,
negative = worse than deterministic.

Approximating with the ledgers' median det/eq ratios (per-OD recomputation from the saved JSONs is
the actual recommendation), the transfer ladder reads:

| cell | ratio (as reported) | approximate gap closure |
|---|---|---|
| multi-convoy headline 35-159 (trained) | 1.24x | **~90%** |
| single-convoy headline 33-71 (trained, n=10) | 1.86x | **~83%** |
| gen15 held-out ODs (same graph) | 1.59x | **~56%** |
| gen16 held-out city (Gdansk) | 1.68x | **~52%** |
| gen22 rotation (Istanbul) | 1.88x | **~24%** |
| whole-Kyiv (scale axis) | 1.88x | **~8%** |

Two consequences. (a) **Honesty:** "transfer holds to the hardest hold-out city" (gen22) coexists
with the fact that on roughly half the Istanbul (OD, seed) cells the policy closes no gap at all
(the ledger's own 4/6, 2/6, 3/6 loss_det clause), and the Kyiv row's headline ("1.88x, beats
random") restates as "closes ~8% of the gap a calibrated policy could close". The current PASS
wording is defensible because the bars were pre-registered, but the thesis figure should show gap
closure so no examiner derives this table first. (b) **Opportunity:** the decay curve itself
(90% -> 56% -> 52% -> 24% -> 8% across the distribution-shift ladder) is a clean, honest,
figure-worthy result: *calibration content decays monotonically with transfer distance while
still dominating naive baselines*, which is a better scientific statement than a list of
threshold passes, and it quantifies exactly where fine-tuning or larger training pools would have
to act. This is one afternoon of eval-only recomputation from saved artefacts.

### 3.3 Defender-side intelligence error is untested (the asymmetric-information gap)

The information structure is asymmetric in the attacker's favour on strategy (pattern-of-life:
the attacker best-responds to the defender's mixture) but asymmetric in the DEFENDER's favour on
the world model: every trained defender observes the true vulnerability map and the true K
exactly. Operationally, threat intelligence is noisy; the first question a defence-domain
examiner will ask of a "conditions on the threat map" claim is what happens when the map is
wrong. This is an eval-only row: evaluate the frozen generalist under perturbed observed maps
(multiplicative noise, dropout of the top-k dangerous edges, or an adversarially chosen map
perturbation at fixed budget) while scoring against the TRUE map's oracle BR; plot degradation vs
noise level, with random-init and the true-map policy as anchors. Graceful degradation is a
robustness exhibit; a cliff is a scoping sentence. Either is one figure, and it composes with
§3.1 (the shuffled-map row is the extreme point of this curve). Note the equilibrium itself has a
useful property here (the value moves modestly under map perturbation, per the §3.1 probe), so
the interesting question is precisely whether the POLICY inherits that stability.

### 3.4 The programme is three bespoke games; the thesis needs one unification table

Across the acts the game changes along four axes simultaneously: interception model
(hard single-convoy 33-71 vs soft-band multi-convoy), action mechanics (walk-trie next-hop vs
route-menu select), adversary class (oracle BR / smooth-FP sparring vs learned SAC interdictor vs
gen19's bounded-memory quantal response), and objective (interception probability vs loss-averse
mission failure). Every switch is individually justified in a ledger, but nowhere does one table
present the game family and which act instantiates which variant. Without it, 12,000 words of
prose will read as a sequence of ad-hoc games; with it (one Methods table: rows = acts, columns =
the four axes plus yardstick), the same material reads as a systematic exploration of one game
family. This is purely a writing artefact but I flag it here because it changes how every result
lands.

### 3.5 Chronicle and doc-web state (small, blocks the FAR)

`SACRED_PROGRESS.md` currently ends at entry 21 (through gen19); gen20-gen23 and the 2026-07-11
expansion completion have no chronicle entries, and the entry numbering around 16-19 is
non-monotonic (documented as intentional in one case, accidental-looking in another). The Final
Activities Report (30 July) is meant to be written from the chronicle. One hour of appending
entries 22-23 and one banner-consolidation pass on HANDOVER (now seven stacked banners deep)
before writing starts will repay itself; this extends CRITIQUE_EXPANSION §1's chronicle finding
to the post-gen19 state.

---

## 4. Logical and argumentative errors (second pass; the examiner critique's §3 list stands)

I re-derived CRITIQUE_EXAMINER §3's ten-item list against the primary sources and concur with all
ten (the over-read gen05 anecdote; the "reactivity is useless" over-generalisation; gen06's
pooled-vs-seed-level significance wording; the gen22/21/20 verdict inflation; the best-is-early
pattern needing one plain paragraph; the gen19 register separation; the headline-instance
migration prevalence sentence; the d3_gdansk reliability gate; the SDVRP title concession). Three
additions:

**4.1 The gen21 causal claim conflates two differences.** The vanilla generalist differs from the
adversarial one in BOTH the presence of an adversary and the objective (travel cost vs mission
failure). "Adversarial training is causal for transfer" is therefore supported against the
cost-objective alternative only. The missing intermediate control is a **domain-randomisation
generalist**: same mission-failure objective, interdictor sampled UNIFORMLY at random each sortie
(threat exposure without best-response pressure). If DR transfers markedly worse than smooth-FP
training, the claim sharpens to "best-response pressure, not mere threat exposure, is causal",
which is the RARL-relevant statement; if DR ties, the honest claim is "threat-aware training"
and that is worth knowing before an examiner asks. One seed, gen21-scale (~1-1.5 h), plus n=2
more vanilla seeds per the examiner's 4.3.

**4.2 "The equilibrium is a reproducible transient" needs its theoretical anchor stated as
scope.** Four failed hold-the-tail attempts justify "inherent to the FP family as instantiated
here"; they do not justify "inherent" simpliciter, and the learning-in-games literature contains
the exact positive counterparts (optimistic/extragradient methods with last-iterate guarantees;
averaging interpretations of FP). CRITIQUE_EXAMINER 3.6 makes the disclosure point; I add the
wording rule: everywhere "inherent" appears, scope it to "inherent to last-iterate
(smooth) fictitious play", cite the known non-convergence of last-iterate FP, and name
optimistic dynamics as the family deliberately left to future work. This converts a potential
"you didn't try the known fix" attack into a scoped design decision.

**4.3 The multi-seed asymmetry between arms is never stated in one place.** SACRED rows get
best-checkpoint selection with n=3-10 seeds; several control rows are single-seed or best-of-one
(gen21 vanilla n=1; vanilla best-checkpoint n=3 on one instance; ALNS deterministic). The
individual disclosures exist, but the thesis should carry one table footnote standard ("every
arm's selection privilege and n") so no reviewer has to reconstruct symmetry case by case. This
generalises CRITIQUE_PREFREEZE §3.5's vanilla-selection point to the whole ladder set.

---

## 5. Methodological oversights (new; prior critiques' lists remain in force)

1. **The shuffled-map transfer row** (§3.1): the single most informative unrun eval in the
   programme. Decides the mechanism wording of the flagship act.
2. **Gap-closure restatement of every transfer number** (§3.2): eval-only recomputation from
   saved JSONs; produces the decay figure and disarms the thin-headroom objection.
3. **Intel-noise robustness curve** (§3.3): eval-only; the operationally-first question.
4. **A retrieval baseline for the ZST act:** nearest-training-instance equilibrium lookup
   (match a held-out OD to the most similar training instance by the F3 feature vector, play that
   instance's equilibrium mixture mapped onto the new menu by route rank). If naive retrieval
   matches the generalist's 1.68, the GNN is doing little beyond feature matching; if the
   generalist wins clearly, the claim strengthens. Eval-only, hours, and it complements the
   LP-distillation control (retrieval = amortisation without generalisation; distillation =
   generalisation without interaction; SACRED = interaction).
5. **The uniform-stack row on held-out cities:** CRITIQUE_EXAMINER measured uniform-stack on the
   two headline instances; the same row on the gen16/gen22 held-out ODs (pure oracle arithmetic)
   completes the transfer ladder's naive-randomisation bound: "zero-shot SACRED beats
   stack-on-a-random-route on unseen cities" is the cheap sentence that pre-empts the cheap
   attack.
6. **Dependence structure in transfer statistics:** the 6 per-city held-out ODs share one policy
   and one graph, so "17/18 cells" language quietly treats dependent cells as independent
   evidence. Keep per-seed means as the inference unit (as the CIs already do) and caption the
   cell counts as descriptive.
7. **Ethics/framing sentence:** the maps are real cities (Kaliningrad, Kyiv, Gdansk, Istanbul,
   East London) in a military-logistics framing. One sentence stating that map data is used
   purely as road-network geometry, with no operational data, keeps the examiner's mind on the
   science. (The guidance PDF's plagiarism/AI rules are already tracked; this is the remaining
   presentational risk.)

---

## 6. Future paths (ranked; engaging the two named ideas and going beyond them)

Calendar rails: FAR + presentation 30 July; experimental freeze 3 August (HARD); thesis + poster
28 August, 12,000 words. Roughly three experimental weeks remain if writing starts inside them.
Every launch remains Kilian's explicit go; each item gets a pre-registered ledger.

### Tier 1: claims-defence (this week; mostly eval-only; do before anything new)

1. **The amortisation-control suite for the ZST act** (the act that carries the thesis):
   (a) **LP-distillation generalist** (CRITIQUE_EXAMINER 4.2; the one new training run; ~1-2
   days); (b) **shuffled-map transfer row** (§3.1; eval-only); (c) **retrieval baseline** (§5.4;
   eval-only); (d) **DR-generalist control** (§4.1; one seed) plus gen21 to n=3. Together these
   bound the generalist from every direction an ML examiner will probe: labels-only,
   geometry-only, memory-only, exposure-only. Whatever survives is the honest claim, and I judge
   the odds good that what survives is strong ("model-free adversarial interaction produces an
   equilibrium amortiser without labels, robust to threat-field decorrelation" if 1b passes).
2. **Gap-closure recomputation + the transfer-decay figure** (§3.2) and the **prevalence figure**
   (CRITIQUE_EXAMINER Tier 1 item 5). Together they answer "did you cherry-pick instances and
   metrics" for the whole thesis in two figures.
3. **Intel-noise curve** (§3.3) and the **held-out-city uniform-stack rows** (§5.5). Half a day
   combined.
4. **d3_gdansk reliability check** (concur with CRITIQUE_EXAMINER 4.4; gate the poster claim on
   it) and the chronicle/doc-hygiene pass (§3.5).

### Tier 2: the differentiators (next 1-1.5 weeks; bounded ledgers; these are what make the work memorable)

5. **The agentic-LLM exploitability benchmark** (Kilian's idea; I endorse it with a sharpened
   design and one warning). The scientific frame that survives review is *calibrated randomisation
   as a capability probe*: the security game has a computable optimum, a computable deterministic
   trap, and a graded ladder in between, which makes it an unusually clean benchmark for language
   agents. Three registers, pre-registered as hypotheses: (a) deterministic ("choose a route"):
   expect ~loss_det; (b) stated-strategy ("output a probability distribution over the menu"):
   score the stated mixture exactly under the oracle BR, the same arithmetic as every other arm;
   expect between uniform and equilibrium, miscalibrated; (c) agentic-sequential (T sorties with
   interception feedback, optionally against the gen19 pattern-of-life adversary where
   history_opt = 0.049 is the computable optimum): the genuinely novel cell, testing whether
   in-context adaptation discovers anti-repeat hedging the way SACRED's window feature did.
   **Design decision to make up front: tool use.** With code execution, a frontier model will
   simply solve the LP (it is a small linear programme) and land at equilibrium, making the
   benchmark a tools demo; the informative registers are no-tools (implicit game reasoning) and
   tools-allowed reported as a separate ceiling row. Pin model versions, temperatures, prompts;
   log transcripts as the reproducibility record; cite the known LLM-randomisation-bias
   literature. Cost: ~1-2 days, eval-only, no training. Payoff: a ladder column and table nobody
   else has, a workshop-paper spin-out, and independent support for the thesis mechanism
   (calibrated randomisation is exactly what language agents lack unaided). **Warning:** in a
   12,000-word thesis it earns one subsection and one table at most; if it threatens the spine,
   it moves entirely to the appendix/spin-out.
6. **The holistic-SBO integration-gap experiment** (Kilian's idea; endorse CRITIQUE_EXAMINER
   item 8, with two sharpenings). Joint design space = placement x hardening x fleet size;
   arm A optimises tier-by-tier (the classical decomposition), arm B one surrogate-guided loop
   over the joint space at matched evaluation budget; primary = the integration gap, with D2's
   L1 = 0.29 tier-coupling as the mechanism story. Sharpenings: make the loop **bi-objective**
   (fleet cost AND exploitability; §1 Obj-4 delta) so the output is a design frontier, not a
   point; and if B3 (cargo values) is ever built, add it to the joint space, because value
   heterogeneity is what makes fleet composition a real variable rather than a count. Run on
   Gdansk with the frozen generalist and it is simultaneously the Obj-4 "simultaneous" sentence,
   the strongest poster exhibit, and the EJOR-shaped spin-out seed.
7. **The risk-aversion spectrum** (endorse CRITIQUE_EXAMINER item 9): sweep the m-of-N /
   CVaR-style loss-aversion parameter at the oracle level and plot the deterministic-vs-mixed gap
   against it. One curve converts the objective-selection story from a defended choice into a
   measured law ("the price of predictability as a function of loss-aversion"), which is the
   single best immuniser against the "you picked the objective where you win" reading.
8. **The multi-OD oracle probe** (endorse item 10): measure the correlation gap (best correlated
   joint mixture vs best product-of-independents) for convoys with different destinations sharing
   corridors. Free, and it decides whether the Tier-3 multi-OD game is worth building. If the gap
   is material, that game is the strongest post-thesis direction (it makes coordination genuinely
   joint and moves the work materially toward the VRP of the title).

### Tier 3: bigger swings (post-freeze or publication phase)

9. **PSRO / double-oracle population training** on a headline instance: upgrades gen20 to the
   principled population loop, connects Obj-3's "population-based" wording a second way, and is
   the standard scaling story for large K (composes with the verified A4 greedy BR).
10. **Optimistic / extragradient last-iterate dynamics**: the algorithm family with last-iterate
    guarantees; the correct first experiment of any follow-on dynamics paper, and deliberately
    out of thesis scope (gen17's gate stays closed).
11. **A history-aware generalist** (merge gen19's window conditioning into the gen16 multi-city
    recipe): one policy, unseen city, adaptive adversary; the "complete SACRED" exhibit and the
    natural flagship of a journal version.
12. **Release the environment + oracles as a benchmark** ("contested-logistics gym"): envs, LP
    and greedy oracles, the city-extraction pipeline, ladder reference solutions, the LLM
    harness from item 5. A day of packaging post-submission; it is the artefact reviewers of
    every spin-out will ask for, and it is how this work gets cited.
13. **Full B1 (Poisson-demand campaign) and B5 (deception/decoys)** remain the recorded
    paper-scale extensions; B5 is where the security-game theory gets deepest (belief
    manipulation), and the latency-vs-predictability coupling of B1 (serving promptly creates
    the pattern the enemy learns) is still, in my view, the most beautiful unbuilt idea in the
    doc web.
14. **A4's K=5 trained cell** stays correctly deferred; the verified matrix-free greedy BR plus
    one honest sentence is the right thesis position.

### Publication map (concurring with CRITIQUE_EXAMINER, with one addition)

AAMAS/IJCAI-agents for the interdiction + ZST + co-evolution act (needs the Tier-1 control suite
and the prevalence figure); RLC/TMLR for the flat-vs-peaked landscape + FP-dynamics + transient
account; EJOR/C&OR for the holistic-SBO story (item 6 + the D-chain); a NeurIPS/ICLR workshop for
the LLM benchmark (item 5). **Addition:** the §3.1/§3.2 results decide which ZST story the AAMAS
paper can tell; run the Tier-1 suite BEFORE drafting any abstract, because "transfers across
cities and threat fields, no labels" and "amortises geometry-consistent equilibria" are different
papers.

---

## 7. What I would do, in order (firm recommendation)

Week of 14 July: the Tier-1 claims-defence suite (items 1-4; the only new training runs are the
LP-distillation control and the one-seed DR control; everything else is eval-only), folding
results into the ledgers as disclosed rows. Week of 21 July: items 5 and 6 (the two Kilian-named
differentiators, both bounded), item 7's oracle sweep and item 8's probe in the gaps; draft the
FAR from the repaired chronicle in parallel and hold 30 July sacred. Then NEXT_STEPS item 7 (the
storyline rewrite), absorbing §1-§5 of this file and CRITIQUE_EXAMINER §1-§4, and write to the
freeze with Tier 3 untouched. The two experiments that change what the thesis is allowed to claim
are the LP-distillation control and the shuffled-map row; every day they stay unrun is a day the
flagship act risks being written around a sentence an examiner can delete. Everything else on
this list makes the thesis better; those two decide what it says.

---

*Artefacts of this critique: this file; `scratch/threatmap_geometry_probe.py` and the §3.1
numbers (oracle-only, reproducible in seconds). No training launched; no `src/` changes; suite
untouched.*
