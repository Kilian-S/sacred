# CRITIQUE_16-07-26.md: holistic critique + future-path programme (Fable, 2026-07-16)

> Requested by Kilian 2026-07-16: a fresh, holistic critique of the whole SACRED interdiction
> programme as an expert academic reviewer would grade it (fit against the five research
> objectives read verbatim from the assessed literature review; triviality; logical and
> argumentative errors in the storyline; oversights in approach and methodology), plus a ranked
> brainstorm of future avenues that would make the project more publishable and scientifically
> interesting, explicitly engaging Kilian's named directions (agentic-LLM benchmarking; SBO for
> holistic contested supply-chain planning) without being limited to them.
>
> **Sources:** the complete HANDOVER read order (HANDOVER, NEXT_STEPS_MASTER, both 2026-07-12
> critiques, NEXT_STEPS_11-07-26, REDESIGN_INTERDICTION, THESIS_STORYLINE, SACRED_PROGRESS 1-21,
> ROADMAP, DIRECTION, DIRECTION_EXPANSION, NIGHT_REPORT, CONTEXT, PROBLEM_REDESIGN, SYSTEM, TASK);
> every ledger gen01-gen25 plus a2/a3/a4/a6_a7_a8/b1/b2/b3_b4/b4/d1/d2/d3/d3_gdansk/f3/
> zst_map_robustness/zst_step0; all four earlier critiques (2026-07-02, INTERDICTION, PREFREEZE,
> EXPANSION); the literature survey PDF (§2.1 aim + §2.2 objectives, verbatim) and the guidance
> PDF (rubric weights); the load-bearing code (build_route_set, the oracles, the generalist
> trainer); and **one new oracle-only probe run this session**
> (`scratch/disjoint_baseline_probe.py`, numbers in §1; seconds of CPU, no training, no `src/`
> change).
>
> **Record-keeping incident (disclosed before anything else).** The persistent agent memory
> contains a five-star index entry dated 2026-07-15 ("DISJOINT-BASELINE FINDING: a 2-line
> max-flow heuristic beats EVERY trained SACRED number in every act; read CRITIQUE_15-07-26_B.md
> before citing any Obj-5/ZST claim") whose two referenced files (`CRITIQUE_15-07-26_B.md` and
> the memory file `disjoint-baseline-finding.md`) exist NOWHERE: not in the working tree, not in
> git history, not in the memory directory. Whatever session produced that finding lost its
> artefacts. This critique independently re-derived and re-measured the claim (§1); the probe
> committed with this file is now the reproducible record. Process lesson: a memory index line
> is not a record; the artefact must be committed in the same session that produces the claim.

---

## 0. Verdict in one paragraph

The programme's discipline remains the best I have seen at MSc level (pre-registration, pinned
SHAs, oracle ground truths, disclosed drift, two survived retractions, a bug-fix arc handled
honestly), and the negative campaign, the fictitious-play dynamics account, and the gen19
dynamic-adversary result are genuine contributions. But the evidence base has one structural
hole that changes what most of the positive acts may claim, and it is measured now: **the
candidate-route menu's own construction (`build_route_set`) begins with the max-flow
(edge-disjoint) decomposition, and a two-line naive baseline built on it, "stack the fleet on
one uniformly-random edge-disjoint route", matches or beats every trained SACRED number in
every static act**: it achieves the exact equilibrium on the single-convoy headline (0.167 vs
SACRED's best 0.276), sits at the low edge of the multi-convoy headline's confidence interval
(0.250 vs 0.256 [0.246, 0.266]), and transfers zero-shot to held-out cities at 1.13x their
equilibria with no training, no labels and no graph exposure, versus the generalist's 1.73x and
every other amortiser's 1.55-1.78x. The ladders' existing "uniform" anchors (0.455, 0.442) are
uniform over the PADDED k-shortest menu, whose near-duplicates are precisely what makes naive
mixing look bad; measured against the baseline a competent practitioner would actually write
down, the static security-game acts reduce to "deep RL partially recovers a max-flow
computation". Three things survive intact and become the honest core of the thesis: the
negative campaign, the learning-dynamics account, and the DYNAMIC register (gen19: 0.050
against a pattern-of-life adversary, which no static mixture, heuristic or LP, can approach:
the best static value is 0.147). One new positive regime is measured this session: at K = 3
(the interdiction budget approaching the min-cut) SACRED beats the heuristic (0.661 vs 0.738),
so the boundary "learning pays when the budget approaches the number of disjoint routes" is
real, cheap to map, and is the correct home for a rescued static claim. The thesis remains
defensible, but its centre of gravity must move, and there is still time to move it before the
3 August freeze.

---

## 1. The decisive finding: the disjoint-route baseline (measured, reproducible)

### 1.1 The fact

`build_route_set` (`src/baselines/interdiction_oracle.py:60`) constructs every candidate menu as
**`nx.edge_disjoint_paths` first** (Menger: one route per unit of min-cut), then pads with
k-shortest near-duplicates. The equilibrium places (near-)zero mass on the padded duplicates
(recorded in the gen08 B2 design itself). So the naive strategy any operations planner would
propose, "enumerate the genuinely independent routes, pick one at random per sortie (stack the
fleet on it)", is two lines of NetworkX, and the menu hands it its input.

### 1.2 The numbers (oracle-exact; `scratch/disjoint_baseline_probe.py`, 2026-07-16)

| act | trained SACRED | uniform-disjoint-stack | inverse-vuln variant | equilibrium |
|---|---|---|---|---|
| SC headline 33-71 hard K=1 | 0.362 (B2-P3) / 0.276 (gen10-SC) / 0.310 (n=10) | **0.167 = the equilibrium** | same | 0.167 |
| MC headline 35-159 N=3 K=1 | 0.256 [0.246, 0.266] (n=10) | **0.250** (1.21x eq) | **0.241** (1.17x) | 0.206 |
| MC pre-fix 62-97 | 0.295 (exact) / 0.447 (post-fix) | **0.250** (1.16x) | **0.241** (1.12x) | 0.216 |
| ZST held-out Gdansk (6 ODs) | generalist 1.733; distill 1.555; retrieval 1.676; DR 2.056; vanilla 2.354 | **1.134x eq, beats loss_det 6/6** | **1.024x eq** | 1.0 |
| ZST held-out Istanbul (6 ODs) | generalist 1.880 | **1.145x eq, beats loss_det 6/6** | **1.048x eq** | 1.0 |

The gen12 sweep cells (held-out 35-159), heuristic beside SACRED:

| cell | equilibrium | uniform-disjoint-stack | SACRED best-ckpt |
|---|---|---|---|
| N=3 K=1 | 0.206 | 0.250 | 0.261 |
| N=3 K=2 | 0.412 | 0.494 | 0.500 |
| **N=3 K=3** | 0.604 | 0.738 | **0.661 (SACRED wins)** |
| N=2 K=1 | 0.179 | 0.249 | **0.232 (SACRED wins, thin)** |
| N=5 K=1 | 0.230 | **0.250** | 0.389 |

### 1.3 Why this happens (mechanism, not accident)

At K=1 with m edge-disjoint routes, one interdicted edge touches at most one disjoint route, so
uniform-disjoint-stack is intercepted at ~(1/m) x (that route's worst edge). On hard
interception over disjoint routes the equilibrium is EXACTLY uniform over them (the ledger's
own recorded correction: "the equilibrium is uniquely uniform for every K" on disjoint routes),
so the heuristic is optimal by construction there. On the soft-band instances the equilibrium
tilts mass by inverse vulnerability, which the closed-form inverse-vuln variant recovers to
within 2-5% of the LP. The padded menu changes none of this (the extras carry no equilibrium
mass); what the padding DOES do is poison the "uniform" anchor: uniform-over-menu stacks mass
on shared edges and degrades with menu size (the CRITIQUE_EXAMINER §5.2 table: 0.25 at R=4 ->
0.53 at R=20). That table was read as "the value of calibration grows with the route set"; the
honest reading is "the padded menu makes the wrong uniform look worse": the value of
calibration relative to the RIGHT naive baseline is nearly constant and small (~1.1-1.2x eq) at
K=1.

### 1.4 What it invalidates, what it forces to reword, what survives

**Invalidated as worded:**
- "SACRED beats every uncalibrated strategy class" (gen13/gen14 ladders): false. The strongest
  uncalibrated class member was never in the ladder, and it matches or beats SACRED.
- The ZST act's comparative framing: every trained/label-consuming amortiser in the transfer
  ladder (distill 1.555, retrieval 1.676, adversarial 1.733) sits ABOVE a zero-training,
  zero-label, zero-graph-exposure heuristic at 1.13. "Zero-shot transferable policies that
  standard algorithms cannot achieve" (the aim, verbatim) is contradicted at this instance
  family: a standard algorithm achieves it better.
- The ZST-vs-LP deployment argument ("the LP needs the full instance model at decision time;
  the policy needs only an observation"): the heuristic needs LESS than both (the graph only,
  not even the threat map) and wins. The A2/A3 map-robustness finding (an information-free map
  costs the generalist only +0.09) is CONSISTENT with this: the generalist was approximating
  the geometry-driven disjoint hedge all along, imperfectly.
- A8's prevalence framing: "uniform-stack/eq >= 1.5x on 93% of ODs" is the padded-menu row;
  disjoint-stack/eq is ~1.1-1.2x on the same population, so calibrated mixing beyond naive
  disjointness buys little at K=1 anywhere in the population.

**Survives, and becomes the honest core:**
1. **gen19 (the dynamic register), now the strongest positive result in the programme.**
   Against the pattern-of-life adversary the BEST possible static mixture scores 0.147
   (iid_eq); the disjoint heuristic is a static mixture, so it is bounded by that; SACRED's
   history-aware 0.050 ~ history_opt 0.049 is something NO static object (heuristic, LP
   mixture, or stated strategy) can reach, with a causal control and a worst-case row. This act
   needs no rewording and should carry more weight.
2. **The K -> min-cut boundary (new, measured this session).** At K=3 on 35-159 (m=4 disjoint
   routes) SACRED 0.661 beats the heuristic 0.738: when the interdiction budget approaches the
   min-cut, naive disjointness saturates (the attacker can cover most of the disjoint set) and
   shared-edge calibration genuinely pays. This is ALSO the regime where equilibrium labels and
   exact oracles die (K >= 4: A4's wall), i.e. the one regime where "only self-play can train"
   AND "no naive heuristic suffices" are simultaneously true. The thesis's static claim should
   live exactly there.
3. **B4's multi-OD correlation gap (14.4%)**: the joint mixture beats the best INDEPENDENT
   product, and the disjoint heuristic is an independent product, so the unbuilt multi-OD game
   is a second regime where the heuristic provably cannot follow.
4. **The negative campaign (gen03-06)** and the flat-landscape mechanism: untouched.
5. **The learning-dynamics account** (instance asymmetry, the FP bracket, the reproducible
   transient, identity-vs-semantics, the coordination boundary): untouched as science, but its
   object must be described honestly: it is a measured account of how model-free adversarial
   learning approaches an equilibrium that is (at K=1) trivially computable and
   near-trivially approximable.
6. **The relative controls** (sacred < vanilla; vanilla < random on transfer; gen20's learned
   antagonist at 0.81x oracle; gen23's ERB negative; gen17/18 boundaries): all internally valid.
7. **The SBO stack's loop pattern** (D1 acquisition, B1 joint-vs-sequential): valid as
   optimisation methodology; but D3's "no LP can participate" motivation weakens, because the
   design loop could equally price the heuristic policy, cheaply. The D3 defence must become
   "price whatever policy will actually be deployed", which is honest and still favours the
   framework, but is no longer RL-exclusive.

### 1.5 The instance-design root cause

This is not bad luck; it is circular instance design. The screens selected for high
loss_det/eq ratio (>= 3), which measures the gap between DETERMINISTIC play and the
equilibrium: it says nothing about the gap between naive randomisation and the equilibrium.
On every screened instance the equilibrium support is essentially the disjoint base set the
menu construction itself provides. A game whose solution a heuristic reads off the menu is not
a hard game, however hard it is for SAC to learn. The constructive fix is to screen future
instances by **disjoint-stack/eq ratio** (the heuristic's suboptimality), not det/eq: §6.

---

## 2. Fit against the five research objectives (verbatim), post-finding deltas only

Six critiques have scored these; I record only what this session changes.

- **Obj 1** (zero-sum Markov game): unchanged (met with the declared Stackelberg/oracle-BR/
  repeated-game deltas; gen20 supplies the learned-agent positive). The Korzhyk-style
  interchangeability sentence recommended on 2026-07-12 is still missing from the doc web.
- **Obj 2** (visual, interactive env): unchanged (met environment-side; the interactive
  exhibit remains unbuilt and is still the cheap Block-C win).
- **Obj 3** (SAC + ATLA + ERB): unchanged (met investigatively; gen23's
  solution-concept-mismatch mechanism remains the best completion of the wording).
- **Obj 4** (SBO, "holistic, simultaneous"): B1's actor-contingent integration gap (0%/19.3%)
  stands; add the disclosed sentence that the design loop's RL-exclusivity claim is retired
  (§1.4.7) and the honest value is "price the deployed policy, whichever policy that is".
- **Obj 5** (evaluate vs SOTA metaheuristics + non-adversarial SAC under varied disruption):
  the comparative clauses stand, but the LADDERS must gain the disjoint rows and the
  conclusions may no longer say SACRED beats every naive strategy class. The varied-disruption
  clause gains the K-boundary nuance: SACRED's edge over the best naive baseline exists only at
  K >= m-1 (measured at exactly one cell so far, single seed).
- **The aim-level ZST promise**: as worded ("policies that standard algorithms cannot
  achieve"), currently false at this instance family. The four binding wording rules from the
  2026-07-12/13 re-scope are necessary but no longer sufficient; the act needs the §6 rescue
  (dynamic transfer and/or high-K transfer) or an explicit concession.

---

## 3. Are the findings trivial? (the sharpened answer)

The question now has a two-part answer the thesis must state itself before an examiner does:

1. **The static K=1 security-game acts are, in hindsight, solving a near-trivial problem
   imperfectly.** The equilibrium is milliseconds by LP, its support is the max-flow
   decomposition the menu construction hands over, and a two-line heuristic is within 1.2x of
   it everywhere measured, including zero-shot on unseen cities. What the trained policy adds
   on these instances is negative to nil. Conceding this crisply costs two sentences; defending
   it costs the thesis.
2. **What is NOT trivial, and is genuinely publishable:** (a) the negative campaign with its
   mechanism chain (when adversarial RL cannot help, and why); (b) the learning-dynamics
   account (FP discipline bracket, reproducible transient, representation effects: no LP or
   heuristic produces these); (c) the dynamic register (gen19: exploiting an adaptive
   adversary's own adaptation, beyond ANY static object, with a computable dynamic optimum);
   (d) the measured boundary where learning first beats the best naive baseline (K -> min-cut,
   new); (e) the honest amortisation taxonomy from Block A (label-free vs label-consuming
   trainers), which becomes MORE interesting with the heuristic row, not less: the ladder now
   reads heuristic 1.13 < distill 1.56 < retrieval 1.68 < adversarial 1.73 < random 1.99 <
   vanilla 2.35, and the right conclusion is that at label-available, heuristic-solvable sizes
   NOTHING learned earns its keep: the value of learning must be sought where labels and
   heuristics both fail (high K, dynamics, multi-OD).

---

## 4. Logical and argumentative errors (new ones; the six prior critiques' lists stand)

1. **The "uniform" anchor strawman (the central one).** Every ladder's naive-randomisation row
   mixes over the padded menu. The 2026-07-12 examiner critique measured uniform-stack and
   declared the hole closed ("SACRED beats every uncalibrated strategy class"); it closed the
   wrong hole, because the natural uncalibrated strategy was never in the comparison set. Rule
   for every future ladder: the baseline set must include the strongest strategy a domain
   practitioner could write in an afternoon, and for route interdiction that is max-flow +
   uniform (or inverse-vulnerability) stacking.
2. **"Calibration value grows with the route set" (gen13 ledger, from menu-sufficiency) is an
   artefact.** The equilibrium is menu-stable BECAUSE the extras carry no mass; uniform-over-
   menu degrades BECAUSE the extras overlap. Nothing about calibration grows; the strawman
   weakens. Retire the exhibit or re-plot against the disjoint row (flat).
3. **The transfer act's mechanism story now over-explains.** zst_map_robustness concluded "the
   hedge is geometry-informed and threat-robust; per-edge map-reading is not the mechanism".
   The simpler complete explanation is that all transfer arms are noisy approximations of the
   disjoint-uniform hedge, which is geometry-only by definition. Any storyline sentence
   implying the generalist learned something richer than that needs evidence that does not
   currently exist.
4. **"Beats loss_det on 17/18 cells" reads as strength but is weak in context**: the heuristic
   beats loss_det on 12/12 held-out cells measured today, at better ratios. Beating the
   deterministic optimum is the floor for ANY sensible randomiser, not evidence of calibration.
5. **The B2 LLM benchmark's pre-registered anchors are now incomplete.** Register (b) scores a
   stated mixture against uniform (0.442) and the equilibrium (0.206); a competent LLM that
   simply reasons "pick uniformly among independent routes" lands at ~0.25 and would BEAT
   SACRED's 0.256: the benchmark as pre-registered could embarrass the headline rather than
   support it. Add the heuristic anchor row BEFORE any live run, and treat "does the model
   discover the max-flow argument" as one of the benchmark's explicit questions (it is
   genuinely interesting: it converts the benchmark from a randomisation-calibration probe
   into a structured-reasoning probe).

---

## 5. Methodological oversights (new)

1. **Baseline-completeness was never a pre-registration requirement.** Metrics and bars were
   pre-registered rigorously; the COMPARISON SET never had to argue its own sufficiency. House
   rule to add: every ladder pre-registers, with justification, the strongest known naive
   baseline, and any new ladder inherits every prior ladder's baselines.
2. **The 2026-07-15 record loss** (header note): a claim of this magnitude existed for a day
   only as a memory index line. Commit critique artefacts in the session that produces them;
   an uncommitted finding does not exist.
3. **Screen criteria measure the wrong gap** (§1.5): det/eq selected instances where
   DETERMINISM fails, not where NAIVE RANDOMISATION fails. All existing prevalence statements
   inherit this.
4. **Single-seed sweep cells now carry load.** The K=3 crossover (SACRED 0.661 < heuristic
   0.738) is the linchpin of the rescued static claim and is currently one seed; it needs n=3
   before anything is written on it.
5. **Fleet-cost fairness for the heuristic row**: the disjoint routes include long detours, so
   the heuristic's cost column should be reported beside SACRED's 123.1 and the equilibrium's
   120.8 when the rows are folded in (eval-only, minutes; likely similar, since the
   equilibrium's own support is the disjoint set).

---

## 6. Future paths (ranked; the constructive programme to the freeze)

Calendar rails: FAR + presentation 30 July; experimental freeze 3 August (HARD); thesis +
poster 28 August, 12,000 words. Roughly 2.5 experimental weeks if writing starts inside them.
Every launch remains Kilian's explicit go.

### Tier 0: claims-repair (days; mostly eval-only; decides what the thesis may say)

1. **Fold the disjoint rows into every ladder** (this probe is the machinery; extend with the
   fleet-cost column and 62-97/33-71 K-sweeps): the two headline ladders, the transfer ladder,
   A8's prevalence figure (add a disjoint-stack/eq distribution), the B2 anchors. One day,
   eval-only.
2. **n=3 the K=3 cell** (SACRED vs heuristic on 35-159, the crossover): ~15 minutes of training
   per seed at 3-parallel under the standing config. The single most important small run left.
3. **Re-spine the storyline** (the C4 rewrite, already scheduled) around the surviving core:
   Act I the negative campaign; Act II the dynamics account (with the bug arc); Act III the
   dynamic register (gen19 promoted to the flagship positive); Act IV the boundary map: WHERE
   learning pays (K -> min-cut, multi-OD, dynamics) and where it does not (K=1 static: the
   heuristic concession, stated first, on our terms). The five-objective scoring keeps the
   2026-07-12 deltas plus §2's.

### Tier 1: the rescue experiments (the week after; each one bounded, each pre-registered)

4. **The high-K act (the static rescue; strongest candidate).** Train SACRED at K=3 and K=4-5
   on 35-159-class instances with the A4 greedy BR as sparring partner and disclosed yardstick
   (the (1 - 1/e) interval), with the heuristic row beside it. K=4 is the first cell where
   labels do not exist (LP RAM-infeasible), heuristics saturate, and self-play is the only
   trainer on the board: if SACRED beats the heuristic there, the thesis owns a claim nothing
   else can make: *"trained where neither exact solvers nor naive heuristics can follow"*. The
   A4-core is verified; the trainer wiring was deferred once for regression risk and is now
   justified by necessity (~1-2 days).
5. **The dynamic generalist (gen19 x gen16; the transfer rescue).** One history-aware policy
   (window feature at the head) trained across the multi-city pool, evaluated zero-shot on
   held-out-city pattern-of-life games against each game's computable history_opt and iid_eq.
   No static heuristic can play this game; if it transfers even partially, the aim's ZST
   sentence is rescued in the one register where "standard algorithms cannot achieve" is
   literally true (~2-3 days; the two ingredients are both proven separately).
6. **The multi-OD game (B4-justified; the VRP-title rescue).** Build the N=2 two-destination
   corridor-sharing game (LP-tractable oracle exists from the B4 probe); train one cell. The
   14.4% median correlation gap is exactly the value the heuristic cannot capture (it is an
   independent product by construction). Also moves the game materially toward the SDVRP of
   the title (~3-4 days; gate on a smoke; the gen18 exploration boundary is the known risk).

### Tier 2: the differentiators (as before, sharpened)

7. **B2 agentic-LLM benchmark (now unblocked on the local workbench).** Endorsed, with the §4.5
   amendment: add the heuristic anchor; pre-register "does the model discover independent-route
   reasoning" as a scored question; registers as designed; pinned open-weight models are a
   reproducibility advantage over commercial APIs. Two TODOs stand (Kilian's go; the generic
   base-url path). Worth at most one subsection + one ladder column in the thesis; its natural
   home remains a workshop spin-out, and its scientific frame improves with the heuristic row
   (calibration AND structural reasoning probe).
8. **Holistic SBO**: B1 is banked; the remaining cheap win is the bi-objective (cost,
   exploitability) frontier plot over designs and the reworded D3 defence (§1.4.7). No new
   compute needed.

### Tier 3: post-freeze / publication register (recorded, not scheduled)

PSRO/double-oracle; optimistic/extragradient last-iterate dynamics; deception/decoys (B5);
full B1 Poisson campaign; the contested-logistics gym release (which now NEEDS the heuristic
reference row to be taken seriously as a benchmark); the publication map from the 2026-07-12
critiques stands, with the note that the AAMAS-shaped paper's honest positioning is now "when
does learning beat the max-flow heuristic in network-interdiction games" (a sharper, better
paper than the one previously sketched).

---

## 7. What I would do, in order (firm recommendation)

This week: Tier 0 items 1-3 (fold the rows, n=3 the K=3 cell, start the re-spine), then
pre-register item 4 (high-K) and item 5 (dynamic generalist) and run whichever Kilian funds
first: my ordering is 4 before 5 (it defends the existing multi-convoy act rather than opening
a new one, and its machinery is verified). Week of 21 July: the funded rescue act(s), B2-live
in the gaps (it is eval-only and the workbench is free), FAR drafted from the chronicle and
held sacred for 30 July. Then C4/writing to the freeze. Items 1-3 are not optional under any
schedule: every day the ladders stand without the disjoint rows is a day the thesis's central
comparative claims are citable in a form a competent examiner can refute with two lines of
NetworkX: and, having now been the third reader to derive those two lines (after the lost
2026-07-15 session and, presumably, any examiner who looks), I would treat that risk as a
certainty.

---

*Artefacts of this critique: this file; `scratch/disjoint_baseline_probe.py` (+ the §1 numbers,
oracle-only, reproducible in seconds; includes the gen12-sweep extension). No training launched;
no `src/` changes; suite untouched.*
