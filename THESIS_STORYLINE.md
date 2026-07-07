# THESIS_STORYLINE.md: the argument, objective by objective (opened 2026-07-06)

> **Purpose.** The narrative spine of the written thesis, argued from the five research
> objectives of the assessed literature review (§2.2) plus the aim's zero-shot-transfer promise.
> `SACRED_PROGRESS.md` is the chronological chronicle (what happened); this file is the
> **argument** (why it happened in this order and what it proves). Primary consumer: Kilian and
> the thesis-planner instance in `../../thesis/`. Citable numbers come from the
> `experiments/genNN_*.md` ledgers only. **Living document**: Act IV (interdiction) updates as the
> gen08 build lands; see `REDESIGN_INTERDICTION.md` for the current direction.

## The promise (March 2026, literature review)

**Aim:** develop and evaluate an adversarial, coevolutionary DRL framework (SACRED) generating
robust, computationally efficient routing policies for the SDVRP.

**Objectives:** (1) formulate the SDVRP as an asymmetric zero-sum Markov game with discrete
action spaces for a protagonist dispatcher and an environment-altering antagonist; (2) build a
visual, interactive simulation environment; (3) develop SAC + ATLA, and investigate ERB
bootstrapping via population-based metaheuristics; (4) incorporate SBO for facility
location/fleet composition; (5) evaluate performance and resilience against SOTA adaptive
metaheuristics **and a non-adversarially trained SAC baseline** under varied network disruption.
The review's identified gaps: SAC unexamined in SDVRP; asymmetric coevolutionary
dynamic-alteration adversaries unstudied in last-mile settings; ZST as the crown jewel.

## The through-line in one paragraph

The thesis begins by betting that adversarial co-training confers robustness on a learned
dispatcher. It builds the machinery (Obj 1-3), discovers that the naive problem formulation is
unlearnable, redesigns the problem to make decisions consequential, chases and retracts a false
positive, and then, properly instrumented, runs the pre-registered experiment the objectives
actually name (Obj 5) and gets a definitive negative with a complete mechanism chain: the learned
adversary cannot learn to attack; the protagonist cannot learn decision-dense arenas; and even in
the best case, constant adversarial exposure degrades the very learning it is meant to harden.
Rather than stopping at the negative, the thesis asks what question minimax training is actually
built to answer, and finds it: worst-case performance against a strategic, adaptive opponent,
measured as exploitability. In contested-logistics applications this is the operationally correct
question (pattern-of-life adversaries), classical deterministic dispatch is structurally weak at
it (predictability is ambushability), and max-entropy SAC's signature feature (calibrated
stochasticity) is precisely the mechanism the solution concept demands. Pursuing that question to
its root reveals *why* the whole campaign failed: the adversary was the wrong KIND. Congestion is
observable, reroutable, and reversible, so a reactive dispatcher captures the value and the attack
landscape is flat (a best-response attacker, even with every fix, cannot beat random blocking).
The thesis then makes its constructive move: change the adversary to the one the application
actually poses, **interdiction** (hidden, irreversible, pre-committed), which is a Stackelberg
security game. There a deterministic router is provably, maximally exploitable and the minimax
**mixed-strategy** router provably robust, so adversarial minimax training wins *by construction*,
with SAC's entropy as the mechanism and a computable equilibrium as ground truth. The final result
is positive and proven at the equilibrium level on the real network (deterministic routing 100%
intercepted, mixed 17-33%). Every act is pre-registered and reported: the thesis is an honest
account of when adversarial training fails, why (the adversary's game structure), and the problem
class where it demonstrably works.

## Act I: the naive instantiation hits the learnability wall (Jun 2026)

The original OSM formulation (diffuse demand, 290 nodes) trains stably after real fixes (alpha
sign, gradient clipping, batched GNN updates) but cannot learn: per-step decisions are
near-inconsequential (Q_Spread collapse 5.3→0.46, delivery pinned ~0.91), and the adversary is
toothless. Lesson that shapes everything after: *stable training and learnable problem are
different properties* (Obj 2/3 machinery validated; problem redesigned via
`PROBLEM_REDESIGN.md`). The curriculum rungs then produce three "near-washes" against reactive
greedy (Stage 0 routing; static-3b assignment, including the retracted "first win"; gen02 dynamic
assignment), which seed the methodology: pre-registered decision metrics, seeds, ledgers,
best-checkpoint discipline, and the retraction culture the examiners should read as rigour.

## Act II: measured properly, the premise fails, and we learn why (gen03-gen06, Jul 2026)

The 2026-07-02 critique (`CRITIQUE.md`) exposes the framing drift ("beat greedy" was never the
objective; Obj 5's named control had never been run) and the protocol biases in both directions.
The reframed, pre-registered campaign then delivers, in order: **gen03**: ATLA confers nothing;
the learned adversary attacks *worse than random* while a 40-line scripted heuristic is 3-6x
stronger; the binding constraint is adversary competence. **gen04**: full motion observability
does not fix it; entropy pinning + reward SNR + γ-myopia diagnosed; co-evolution parked by a
pre-registered gate. **gen05**: on the decision-dense hybrid arena neither arm learns at all;
ceiling compression identified as an evaluation pitfall (weak policies fake robustness); one
nugget: against a competent, predictable victim the seeing learned attacker becomes the
strongest attack in the portfolio (+1667 on greedy). **gen06**: in the competence-gated arena the
primary *reverses significantly* (dD_targeted = −881 ± 284, 0/3 pairings; worse even under the
training attacker; zero clean cost): adversarial exposure made the policy worse under aimed
attacks, and the most robust policy measured is the deterministic reactive dispatcher, exactly as
the reactive-dominance literature (Ritzinger et al. 2015) predicts. Unifying mechanism: the
zero-sum latency reward buries each agent's controllable contribution under an uncontrollable
shared queue baseline (with post-hoc telemetry pointing at temperature/entropy mis-calibration
and collapse-regime training distributions as the proximate channels; to be ledgered).

Act II's constructive residue: four (now five) named preconditions for adversarial training in
this domain, and the evaluation methodology itself (pre-registration, competence gates, held-out
attack portfolios, per-policy best responses, paired stochastic evaluation), each clause earned
by a documented failure.

## Act III: the right question (2026-07-06 redirection)

The pivot of the argument, and the moment the thesis stops being only a diagnosis. Two
observations force it. First, *the campaign's own data* contain the seed: the learned attacker's
one success was against the competent, deterministic victim (gen05 +1667): predictability, not
weakness, was the exploitable surface. The post-hoc snapshot sweep (gen06 ledger appendix A3.2)
then found the same axis inside the vanilla arms themselves: aimed-attack robustness *declines*
with clean training time as the policy specialises and commits: competence is purchased with
predictability, which is precisely the currency an adaptive adversary collects. Second, minimax training optimises the worst case by
construction; measuring it by average-case degradation under fixed disruptions (Act II) asks it
to win a game it was never built for, against a reactive baseline the literature says is
near-unbeatable there. The question adversarial training is *for* is: **how badly can an
adversary who has studied you hurt you?** That is exploitability, the standard currency of the
security-games and empirical-game-theory community (AAMAS), and it is the operationally correct
question in the supervisor-motivated application space (contested resupply, asset escort:
adversaries observe repeated operations and adapt). In this register the previous conclusions
invert honestly: greedy's determinism becomes its measurable weakness; SAC's entropy, which Act
II showed *costing* robustness, becomes the mechanism that buys unexploitability; and the
zero-sum game formulation of Obj 1 finally gets evaluated against its own solution concept
(distance from minimax) rather than against a proxy. The gen07 work (branch `gen07-contested`,
`experiments/gen07_contested_matrix.md`) built five learnability fixes and probed the
contested-*destination* arena, and in doing so it delivered the act's real payload: a mechanistic
diagnosis. The corrected best-response gate, with all fixes applied, still lands at 0.35× random,
and the telemetry shows why, the attacker's Q-values across blocks are near-identical because on
a stressed queueing network every route-reach block causes similar cascading damage. **The attack
landscape is flat where it is large and thin where it is differentiated.** This is not an
optimisation bug the fixes can cure; it is a structural property of the congestion adversary, and
it explains gen03/04/06 at one stroke. Act III therefore closes not with a matrix but with a
verdict: adversarial RL cannot win against *this kind* of adversary, and the reason points
directly at the redesign.

## Act IV: the redesign where adversarial RL demonstrably works (interdiction, 2026-07-06 →)

The constructive climax. Change the adversary from congestion to **interdiction/ambush**, the
threat Application 1 (contested autonomous resupply) actually poses: hidden (unseen until struck),
irreversible (interception, not a delay you route around), and pre-committed (positioned before
the sortie, against your pattern). Against such a threat reactivity is useless and the only
defence is anticipation and unpredictable routing, which is the canonical **Stackelberg security
game** (the deployed ARMOR/PROTECT/AAMAS lineage). Its structure guarantees the thesis's claim: a
deterministic router (shortest-path, greedy, a collapsed vanilla-SAC policy) is maximally
exploitable, and the minimax mixed-strategy router provably cuts interception. The elegant
reversal: SAC's max-entropy objective, the very feature that *cost* robustness in Act II, is now
exactly the mechanism that produces the equilibrium mixed strategy. Proven before any training, at
the equilibrium level, on the real Kaliningrad graph: a deterministic route is intercepted 100% of
the time, the mixed route 17-33% (`scratch/interdiction_game_probe.py`; gap 0.67-0.83, tunable in
the enemy's budget). SACRED (SAC entropy + ATLA as iterated best response) learns toward that
equilibrium; shortest-path and vanilla sit at the exploitable extreme; and the equilibrium is
*computable*, giving a ground-truth reference no earlier act had. Pre-registration:
`experiments/gen08_interdiction.md`; build plan: `ROADMAP.md` Phase I (Kaliningrad, single convoy
first). This is the positive result the thesis was always reaching for, reached by choosing the
problem where the mechanism is the solution rather than a forced fit.

**Act IV realised (2026-07-06/07; all numbers in the gen08 ledger).** The build delivered in
four pre-registered steps. The feasibility slice banked the first positive result (interception
100% -> 23%, equilibrium 16.7%). The first asymmetric instance (length-derived vulnerability)
FAILED its sacred-vs-vanilla primary and taught the deepest lesson of the act: when vulnerability
correlates with travel cost, the non-adversarial control IMITATES the equilibrium without
understanding it, so the comparison must be staged where imitation is provably impossible. The
shared-edge instances provide exactly that (the road network's own k-shortest structure: the
oracle proves no cost-driven mixture gets below 2.8x the equilibrium), and there the control is
WORSE than uniform noise: cost-calibrated mixing is predictability with extra steps. Making
SACRED's own policy stable took a measured study in fictitious-play discipline: best-responding
to the latest pure commitment cycles (average converges, iterates orbit), a stale all-history
mixture lets the cost gradient park the policy, and SMOOTH fictitious play (softmax best
response to recent play, temperature probe-pinned) stabilises: whereupon the pre-registered
primary PASSED on every clause: **shortest-path 1.000 > vanilla 0.477 > uniform 0.455 > SACRED
0.362 >> equilibrium 0.167** (trailing-averaged policy, 3 seeds, pooled). Honest boundary,
reported as measured: the strong form (within 0.05 of the equilibrium) was not met (distance
0.163-0.239); closing that last-iterate gap is future work, and the FP-dynamics bracket is
itself a contribution: the thesis can now say not only THAT adversarial training works here but
WHICH realisation of it converges and why.

## Objective-by-objective arc

**Obj 1 (zero-sum game formulation).** Act I formulates it. Act II shows the naive instantiation
is unlearnable on both sides and isolates why (SNR of the coupled reward; entropy machinery vs
flat action spaces). Act IV realises it in its purest form: a **Stackelberg security game** with a
*computable minimax equilibrium*, so SACRED is evaluated against the game's actual solution
concept and its distance-to-equilibrium is measurable. Chapter placement: formulation in Methods;
the flat-landscape learnability result in Results; the security-game realisation in the final
Results act. Met *more deeply than promised*: formulated, characterised (when it fails), and
solved (where it works).

**Obj 2 (simulation environment).** Met by the campaign machinery: problem rungs behind one CLI
(`--problem {osm,stage0,assign,dynassign,hybrid,contested}`), Poisson dynamics, congestion/blocking
physics, SMDP event wrapper, PyGame visualiser, 100+ green tests, ledgered reproducibility. Act IV
adds the **interdiction game layer** (`--problem interdiction`: hidden pre-committed interdiction +
interception reward) on the same graph scaffolding, a targeted game-structure change, not a new
env. Chapter placement: Methods, with the visualiser + the equilibrium-oracle figures.

**Obj 3 (SAC + ATLA + ERB bootstrapping).** SAC and ATLA are present throughout; Act IV is where
they become *load-bearing arguments with a positive result*: max-entropy SAC as a principled
mixed-strategy learner (the entropy IS the equilibrium randomisation), and ATLA as iterated
best-response ≈ fictitious play converging toward the security-game equilibrium (its natural home).
ERB bootstrapping (inconclusive at n=1 in gen01) returns with a proper slot: seed from the
shortest-path baseline or the equilibrium solver, with a pre-registered time-to-competence
ablation. The "population-based metaheuristic" wording is addressed openly (double-oracle over
interdiction strategies is literally a population method).

**Obj 4 (SBO for depots/fleet).** The weakest objective all campaign; Act IV gives it a natural,
novel form: **interdiction-aware base/FOB placement**, site bases for egress edge-connectivity to
minimise equilibrium interception. Evaluate over a placement grid, fit a small neural surrogate
predicting interception/exploitability from the design, validate the surrogate's chosen placement
by full simulation. A genuine metamodel-coupling demonstration ("place bases to minimise
exploitability"). Fallback: descope with supervisor agreement, demonstrator design specified.

**Obj 5 (evaluation vs metaheuristics and vs non-adversarial SAC under varied disruption).** The
objective that drove the Act-II reframe (its control had never been run) and that Act IV
completes with a POSITIVE result. Non-adversarial SAC (vanilla): the core control, sitting near
the exploitable extreme in Act IV. Metaheuristics/classical: shortest-path routing is the genuine
operational default and is *provably* maximally exploitable (100% intercepted at the equilibrium)
, turning the comparison into a headline finding. The **equilibrium oracle** is a computable
ground-truth reference no metaheuristic gives: SACRED is scored by how close it gets to minimax.
"Varied levels of network disruption": the interdiction-budget K and edge-connectivity axes,
reported as curves. Chapter placement: the final Results act IS this objective.

**ZST (aim-level promise).** Scoped, honest version: zero-shot transfer of the Act-IV mixed-strategy
policy to a held-out OD pair / theatre, reporting how the interception gap and distance-to-
equilibrium transfer. The mixed-strategy concept is inherently transferable (unpredictability is
graph-agnostic). Fallback: transfer of the *diagnosis* (Act III's flat-landscape mechanism), plus
future work.

## The sceptical-examiner bank (v3: reshaped for the interdiction redesign)

1. *"You changed the problem until RL won."* The change is principled and forced, not
   opportunistic: Act II-III show *mechanistically* (the flat attack landscape, the corrected BR
   gate at 0.35× random) that congestion is structurally the wrong adversary, and interdiction is
   the threat Application 1 actually poses. We moved to the *deployed, canonical* security-game
   structure (Tambe et al.), kept the genuine operational baseline (shortest-path), and validate
   against the *true equilibrium*. Choosing the problem where the mechanism is the solution is
   good science; the full evidence trail is dated and pre-registered.
2. *"Isn't the positive result just a game-theory tautology?"* The equilibrium says a gap EXISTS;
   the thesis's contribution is that a *deep-RL* agent (SAC + ATLA), with no explicit game solver,
   *learns* toward that equilibrium from experience on a real road network, and that its
   max-entropy objective is the natural mechanism, plus it scales past where the LP oracle is
   tractable (large graphs, multi-convoy). The oracle is the yardstick, not the method.
3. *"Your survey says reactive is near-optimal; why did you fight it?"* We did not: Act II
   documents reactive-dominance for *congestion*, and Act IV moves to interdiction, where
   reactivity is structurally useless (the ambush is set before you move) and anticipation is the
   only defence.
4. *"Isn't the mixed strategy just adding noise?"* No: it is the *calibrated* equilibrium
   randomisation. Vanilla SAC (uncalibrated / collapsing to determinism) and shortest-path are the
   controls; SACRED approaches loss_mixed while they sit at loss_det. The distance-to-equilibrium
   is measured, not asserted.
5. *"Four of five objectives had no results."* Each now has a positive result or a scoped
   demonstrated form on the interdiction game (see the arc): Obj 1 solved against its equilibrium,
   Obj 5 the headline gap, Obj 4 interdiction-aware placement, Obj 3 ATLA-as-fictitious-play, ZST
   the transferable mixed strategy.
6. *"Statistical rigour?"* The gen07 methodology carries over: pre-registration, held-out
   instances, paired comparisons, dual-level significance, gating expensive training on cheap
   probes (and here, on the computable equilibrium).
7. *"Was the campaign wasted?"* No: it is the rigorous negative that *motivates and justifies* the
   redesign, and the evaluation methodology is a standalone contribution. The thesis is honest
   about when adversarial RL fails and why, then shows where it works.

## Candidate one-sentence thesis statements (drafts, Kilian to choose tone)

- *Adversarial reinforcement learning does not make a congestion-facing dispatcher robust (we show
  mechanistically why: the attack landscape is flat), but for the interdiction threat of contested
  logistics it learns mixed-strategy routing policies that approach the security-game equilibrium
  and are far less exploitable than shortest-path or non-adversarial RL.*
- *When does adversarial training help routing? Not against reroutable congestion (a rigorous
  negative), but against hidden, committed interdiction, where SAC's entropy becomes the
  equilibrium mixed strategy and adversarial RL provably wins, demonstrated on a real road network
  against a computable optimum.*
