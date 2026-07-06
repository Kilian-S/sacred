# THESIS_STORYLINE.md: the argument, objective by objective (opened 2026-07-06)

> **Purpose.** The narrative spine of the written thesis, argued from the five research
> objectives of the assessed literature review (§2.2) plus the aim's zero-shot-transfer promise.
> `SACRED_PROGRESS.md` is the chronological chronicle (what happened); this file is the
> **argument** (why it happened in this order and what it proves). Primary consumer: Kilian and
> the thesis-planner instance in `../../thesis/`. Citable numbers come from the
> `experiments/genNN_*.md` ledgers only. **Living document**: Act IV updates as gen07 lands.

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
stochasticity) is precisely the mechanism the solution concept demands. The final experiment
(gen07) tests whether adversarial co-training, with the five pathology fixes derived from the
diagnosis, buys a better efficiency-versus-unpredictability frontier than deterministic dispatch
and naive randomness. Whatever gen07 shows, every act is pre-registered and reported: the thesis
is an honest account of when adversarial training fails, why, and under what conditions it can
work.

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
weakness, was the exploitable surface. Second, minimax training optimises the worst case by
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
(distance from minimax) rather than against a proxy.

## Act IV (planned): the conditions under which it works (gen07)

Pre-registered before building (see `DIRECTION.md` §5, `ROADMAP.md`): contested-resupply arena;
arms {vanilla, DR, entropy-matched vanilla, SACRED with curriculum + adversary population};
primary = the exploitability gap under per-arm tailored attack portfolios (learned best response
AND fitted scripted attacks, take the max); secondaries = the Act-II held-out portfolio
(continuity), clean premium, and the efficiency-versus-exploitability frontier with budget-axis
curves. The five fixes (curriculum exposure, counterfactual rewards, entropy repair, factored
attacker + population, credit horizon) are exactly the Act-II diagnosis turned into design, so
gen07 doubles as the *test of the diagnosis*: if the preconditions were correctly identified,
supplying them should produce the benefit where it is structurally available. Both outcomes are
writable: success gives the constructive final act; a null gives a sharpened impossibility
narrative with the fixes ruled out as sufficient.

## Objective-by-objective arc

**Obj 1 (zero-sum game formulation).** Act I formulates it. Act II shows the naive instantiation
is unlearnable on both sides and isolates why (SNR of the coupled reward; entropy machinery vs
flat action spaces). Act III/IV evaluate the game against its own solution concept:
exploitability as measured distance from minimax, with a formal note that the
counterfactual-baseline rewards preserve the equilibrium (the subtracted twin term is constant in
both agents' actions). Chapter placement: formulation in Methods; the learnability conditions in
Results/Discussion. The objective is met *more deeply than promised*: not just formulated, but
characterised.

**Obj 2 (simulation environment).** Met by Act I-II machinery and unchanged since: five problem
rungs behind one CLI (`--problem {osm,stage0,assign,dynassign,hybrid}`), Poisson dynamics,
congestion/blocking physics, SMDP event wrapper, PyGame visualiser, 83 green tests, ledgered
reproducibility. Act IV adds the contested skin (naming + curriculum module), no new physics.
Chapter placement: Methods, with the visualiser figures.

**Obj 3 (SAC + ATLA + ERB bootstrapping).** SAC and ATLA are present throughout; Act IV is where
they stop being merely present and become *arguments*: max-entropy SAC as a principled
mixed-strategy learner (entropy-regularised equilibria), ATLA upgraded with an adversary
population (fictitious-play flavour, already cited in the review). ERB bootstrapping, inconclusive
at n=1 in Act I (gen01), returns with a proper slot: the curriculum needs a competent starting
policy, so demo-seeding gets a pre-registered time-to-competence ablation. Honest adaptation to
report: demos come from the dynamic dispatcher (greedy insertion / rolling variant), because
static population-based solvers do not fit a dynamic stream; the objective's "population-based"
wording is addressed openly (rolling-ALNS demos if funded, else the taxonomy note).

**Obj 4 (SBO for depots/fleet).** The weakest objective all campaign (untouched; descope was on
the table). Act IV gives it a defensible reduced form: **robust facility location**: evaluate the
trained arms and attack portfolio over a coarse depot-placement grid, fit a small neural
surrogate predicting performance *and exploitability* from the design, select the surrogate's
best design, validate by full simulation, report surrogate accuracy. Scoped as a demonstration of
the metamodel coupling (the objective's core idea), not a full SBO loop; that distinction is
stated in the thesis. Fallback: descope with supervisor agreement, framed as future work with the
demonstrator design already specified.

**Obj 5 (evaluation vs metaheuristics and vs non-adversarial SAC under varied disruption).** The
objective that drove the Act-II reframe (its control had never been run) and the one the
redirection completes. Non-adversarial SAC: the core control in both Act II and Act IV, now
joined by DR and entropy-matched controls that make the causal attribution clean. Metaheuristics:
greedy insertion as the strong reactive reference throughout; rolling-ALNS as the SOTA-adaptive
representative in Act IV (eval-only), where its determinism is predicted to make it *measurably
exploitable*: the comparison becomes a finding about classical optimisers in adversarial
settings, not a box-tick. "Varied levels of network disruption": the budget axis reported as
curves in both registers (average-case D and exploitability). Chapter placement: the results
chapters ARE this objective.

**ZST (aim-level promise).** Scoped, honest version: one zero-shot transfer of the final Act-IV
arms and attack portfolio to a held-out geometry (evaluation only), reporting how competence,
exploitability and the frontier position transfer. If time forbids even that, the fallback
recorded in Act II stands (transfer of the *diagnosis*), plus future work.

## The sceptical-examiner bank (v2: maintained; answers planned or in hand)

1. *"Your survey says reactive is near-optimal; why did you fight it?"* We did not, twice over:
   Act II's control comparison is RL-vs-RL (greedy is a reference line), and Act III moves to the
   register where reactive determinism is the weakness. The survey's reactive-dominance claim is
   about disruption response, not strategic opposition.
2. *"You changed the question after losing."* Pre-registered redirection, recorded in
   `DIRECTION.md` dated 2026-07-06 with supervisor sign-off sought BEFORE any gen07 build; the
   Act-II negative is reported in full as a headline result, not buried.
3. *"Your exploitability metric depends on your attacker's competence."* Portfolio-max rule:
   exploitability = max over the learned best response AND scripted attacks fitted per victim;
   Act II documented exactly this failure mode and the metric is designed around it.
4. *"Isn't this just adding noise?"* The entropy-matched vanilla control, and the frontier
   framing: the claim is cheap unpredictability, not unpredictability.
5. *"Zero-sum plus a baseline is not zero-sum."* The twin-rollout term is constant in both
   agents' actions per episode; strategy-equivalence preserved (formal note in Methods).
6. *"Instance-level significance with 3 seeds?"* Dual reporting rule from Act II onward: pooled
   instance-level CI (pre-registered) + per-pairing sign consistency + seed-level sensitivity.
7. *"Four of five objectives had no results at the review stage."* Each objective now has either
   a substantive result or a scoped, demonstrated reduced form with the descope agreed and dated
   (see the arc above).
8. *"Why believe the five preconditions caused gen07's result (if positive)?"* The controls map
   one-to-one to mechanisms (DR isolates exposure; entropy-matched isolates stochasticity;
   curriculum ablation if funded); and the Act-II diagnosis predicted them in advance.

## Candidate one-sentence thesis statements (drafts, Kilian to choose tone)

- *Adversarial co-training does not buy average-case robustness in stochastic-dynamic routing
  (and we show why), but it does buy calibrated unpredictability against strategic adversaries:
  we characterise when each claim holds and provide the evaluation methodology for both.*
- *From "does adversarial training make routing robust?" to "what is adversarial training for?":
  a pre-registered diagnosis of failure and a demonstration of the conditions for success in
  contested logistics.*
