# CRITIQUE.md — Fresh-eyes critique of the SACRED approach (Fable, 2026-07-02)

> Requested by Kilian via `HANDOFF_PROMPT.md`: interrogate the whole approach before continuing.
> Sources: the literature review PDF, `CONTEXT.md`, `PROBLEM_REDESIGN.md`, `SYSTEM.md`, `TASK.md`,
> the geometry plots, the core code (`smdp_wrapper`, `graph_env`, `sac`, `networks`, `sacred_atla`,
> `transition_builder`, factories, baselines, eval scripts, headroom probes), the `experiments/`
> ledgers, and two new probes (`scratch/critique_probes.py`, output below). Test suite re-run:
> **75 passed** (matches the record). No `src/` code was changed.

---

## 0. Verdict in three sentences

The machinery is real and the engineering discipline is now good, but the project has spent three
rungs chasing the **wrong headline**: "adversarially-trained RL beats a reactive classical
dispatcher" is neither the thesis's stated research objective nor a claim the literature predicts
is winnable at meaningful effect size — the thesis's own survey (Ritzinger et al. 2015) reports
reactive methods capture nearly all the value and anticipation adds ~0–5%. Meanwhile the one
comparison the research objectives actually name — **SACRED vs a non-adversarially-trained SAC
baseline under disruption** (Obj. 5, the RARL robustness claim) — **has never been run in any
rung**. The three "near-washes" are additionally unreliable as evidence because the evaluation
protocol carries biases in both directions and the learned agents fight greedy with strictly less
information, a myopic horizon, and (in the untrained hybrid rung) two outright mechanical defects
found by probe — so the honest conclusion is not "RL can't beat greedy," it is "the current
experiments could not have resolved the question, and the question itself should be demoted."

---

## 1. The framing drift (the core critique)

The lit review's research aim: *"develop and evaluate an adversarial, coevolutionary DRL framework
to generate **robust**, computationally efficient routing policies"*, with Obj. 5 explicitly:
*"Evaluate the performance and **resilience** of the SACRED framework against SOTA adaptive
population-based metaheuristics **and a baseline non-adversarially SAC-trained agent** under
varied levels of network disruption."* The headline scientific claim is **robustness from
adversarial training** (RARL: Pinto et al. 2017 — adversarial training improves robustness to
train/test shift), plus ZST as the crown jewel.

Somewhere between the pivot and Stage 0, the success criterion mutated into **"RL beats greedy
under attack"** (`gap_atk < 0`). Every rung, every ledger, every retraction since has been about
that number. Two things are wrong with this:

1. **It is not the thesis claim.** A protagonist that merely *matches* a strong reactive
   dispatcher but *degrades far less under attack than a non-adversarially-trained policy* fully
   validates Objectives 1–3+5. The control that makes that claim measurable — vanilla SAC (no
   adversary) and ideally random-perturbation SAC (domain randomization) — appears nowhere in
   `experiments/`, nowhere in the eval scripts, nowhere in the roadmap.
2. **The literature the thesis cites predicts the wash.** Ritzinger et al.: reactive ≈ +10% over
   static planning; anticipatory ≈ 0–5% *on average* over reactive. The project's greedy baseline
   is reactive **with perfect global congestion observability and exact Dijkstra re-planning at
   every edge** — near the top of the reactive class. Repeatedly observing "RL ≈ greedy ± noise"
   is what the surveyed literature says should happen. Three near-washes are not a machinery
   failure signal; they are a *replication* of the field's central finding.

**Consequence:** "beat greedy" should be demoted from success criterion to *reference line*, and
the missing non-adversarial control promoted to the headline comparison.

## 2. The three "near-washes" are not valid evidence for the structural conclusions drawn

The conclusions "assignment-only is too thin" / "destination mode starves the antagonist" /
"next-hop routing is the missing lever" treat the washes as informative. The protocol cannot
support that:

- **(a) Fixed-adversary asymmetry (anti-RL bias).** The "fixed final antagonist" is the
  co-evolved net — an approximate best response **to the RL protagonist**. Greedy is then scored
  against an adversary that was never optimized against greedy. `gap_atk` therefore compares
  RL-vs-its-own-nemesis against greedy-vs-an-off-target-attacker. A fair robustness comparison
  needs per-policy best-response adversaries or a common *portfolio* of attackers (random,
  scripted-targeted, learned-vs-greedy, learned-vs-RL), reported per attacker.
- **(b) Best-of-snapshots selection on the test adversary (pro-RL bias).** `select_best_checkpoint`
  takes the min over ~15–30 snapshots of the same quantity used as the verdict. The gen02 ledger
  admits this ("selection-biased; seed0's best being untrained ep50 is the tell"). Selection must
  happen on a validation attacker, reporting on a held-out one.
- **(c) Deterministic-argmax eval of a max-entropy policy.** `deterministic=True` for both agents
  collapses the SAC policy to its mode. If the value of adversarial training is partly a *mixed*
  (unpredictable) strategy — exactly the recovery mechanism the hybrid rung hypothesizes — the
  eval erases it by construction. Stochastic eval, several rollouts, averaged, is the matching
  protocol (and it also fixes the knife-edge fragility of single deterministic episodes).
- **(d) Statistical power ~ zero at the chosen operating point.** dynassign runs at ρ≈1 — the
  regime where total-latency variance is maximal (heavy-traffic queueing). Eval std ±~1000 vs
  plausible effects of ±100–300 → needs ~50–100 paired instances, not 5; and 2 training seeds.
  Common-random-numbers pairing (same demand seed, same attack realization for both policies) is
  only partially exploited.
- **(e) Q "runaway" may be mis-read.** With reward = −(outstanding units) per tick, an antagonist
  Q of ~116 at scale 0.1 under a compounding queue is roughly what a *correct* critic should
  output (its per-decision reward legitimately grows with the backlog). Some of the "instability"
  narrative is plausibly the critic tracking a non-stationary but real quantity. The co-evolution
  cycling is real, but it is a *protocol* problem (when you measure), not necessarily divergence.

**And separately — the learned side fought handicapped in every rung:**

- **(f) Information asymmetry.** Greedy uses exact congestion-aware Dijkstra over the whole graph
  at every decision. The policy is a 2-layer GATv2: its receptive field is 2 hops on a graph of
  diameter ~44. Outside dynassign (which added ETA features), the policy cannot see a blockade —
  or the demand geometry — beyond 2 edges away. The hybrid rung as built has **no ETA features**
  (`truck_etas` is dynamic-mode-only), so it would route with 2-hop sight against an adversary
  whose whole point is placing blocks far ahead. You cannot out-anticipate an opponent you cannot
  see.
- **(g) γ-myopia against the anticipation hypothesis.** γ=0.99 **per tick** with round trips of
  50–100 ticks and episodes of 800–1500: 0.99^300 ≈ 0.05. The queue-compounding payoff that
  motivates the whole redesign lives beyond the discount horizon. The agent is structurally
  myopic in exactly the dimension the design wants it to be far-sighted.
- **(h) The hypothesized "timing" strategy is outside the action space.** There is no wait/no-op:
  an unassigned truck with ≥1 pending request *must* be assigned; an assigned truck at a branch
  *must* move. "Wait out the block at the depot" — the most natural anticipatory counter — is
  illegal.
- **(i) Hybrid observation cannot express the task.** `assigned_target` is not in `observe()` and
  not featurized. When routing, the policy cannot see *its own goal* (only the corridor mask hints
  at it); when assigning, it cannot see other trucks' commitments (`targeted_by_other` keys off
  `destination`, which in hybrid mode is just the next adjacent node). Mid-edge trucks are
  invisible in node features for both agents. The antagonist also never sees truck commitments
  (its featurization is called without `active_truck_id`), so its "anticipation" is carried
  almost entirely by the reach mask, not by learned understanding.

Items (a)–(e) mean the wash verdicts are unreliable in both directions; items (f)–(i) mean that
even a real edge would have had trouble materializing. Both can be true at once — and both point
away from "run H7 as-is" and away from "the lever must be routing."

## 3. New probe results (scratch/critique_probes.py — light CPU, no training)

**Probe A — the hybrid rung has a mechanical defect that fires every episode.** Sequential
claiming is per-decision-event only; nothing prevents a truck being assigned to a request another
truck is already en route to (cross-event). When the other truck serves it first, the loser
arrives at a zero-demand node and — because `assigned_target` only clears on serve or reload —
**orbits it forever**:

```
[greedy no-attack]      delivered=8/8  ticks=1500  last_serve_tick=246
   multi-assigned nodes={'130': 2}   trucks stuck on zero-demand target at end: [(0, '130')]
[greedy vs route-reach] delivered=8/8  ticks=1500  last_serve_tick=439
   multi-assigned nodes={'17': 2}    trucks stuck on zero-demand target at end: [(1, '17')]
```

Every hybrid episode (both cells of H5 included) ends with a zombie truck, runs to the full
1500-tick horizon (~5× the useful span → ~5× wasted wall-clock per training episode), and — in
training — would flood the replay buffer with hundreds of zero-reward orbit transitions per
episode, drowning the ~50 meaningful decisions. **H7 as-built would have been poisoned. Not
running it was the right call.**

**Probe B — the "+79% recoverable headroom" is mostly not recoverable.** Permanently blockading
just the single `('0','1')` gateway costs greedy **+48.7%** (902→1341) — pure unavoidable detour,
no policy that must reach this demand cluster avoids it while the block stands, and the antagonist
has ~6× the budget needed to keep it standing. So of H5's +79 points of "attack cost = max
recoverable", roughly **49 points are floor**, leaving ≤ ~30 points against an adversary no
smarter than "hold one gateway". Worse, with 5 gateways blocked greedy fails outright
(+179%, 7/8 delivered) — and the budget (4000 ≈ 32 sustained blocks) supports 2–5 concurrent
sustained blocks. **At this budget the equilibrium of the arms race is plausibly "everyone gets
crushed", where gaps between policies are noise.** The +79% figure also came from a scripted
attacker (`sorted(mask)[0]`) that targets the gateway only by lexicographic accident. The
adversary here is not too weak — it is *overpowered and under-informed*: strong enough to
saturate the demand region's boundary, blind enough (no commitment/motion features) that its
learned version may never do so reliably. Neither extreme yields a meaningful game.

## 4. Is "next-hop routing is the missing lever" supported?

No — and it was already contradicted by the project's own data before the hybrid was built:
**Stage 0 was a pure next-hop rung and washed.** The explanation then ("single truck, so no
assignment headroom") and the explanation after dynassign ("assignment-only, so no routing
leverage") are individually plausible but jointly form a pattern: each wash produces a new
exculpatory factor and a new rung. That is motivated-reasoning risk, and the H5 gate — the +79%
number — does not test recoverability at all (its own text admits this; Probe B now quantifies
that most of it is floor). The honest statement: *routing control gives the **antagonist**
leverage; whether it gives the **protagonist** any is untested and, per Probe B, tightly bounded.*

## 5. What a skeptical examiner would say today

1. "Your survey says reactive ≈ optimal minus ε; your result is RL ≈ reactive greedy. Why is the
   central experiment of the thesis an attempt to refute your own literature review?"
2. "Where is the non-adversarial baseline named in Objective 5? Without it there is no evidence
   adversarial training did anything."
3. "The adversary's budget/reach/geometry were redesigned after each null result. What stops me
   reading the final configuration as selected-to-produce-the-result?" (The §6 answer — sweep the
   axes and report curves — was in `PROBLEM_REDESIGN.md` all along.)
4. "Your fixed-adversary metric pits each policy against an attacker trained on only one of them;
   your best-checkpoint is selected on the test attacker; your deterministic eval mode discards
   the stochastic policy you trained. Which sign is `gap_atk` biased in? You don't know."
5. "SBO, ERB-from-metaheuristics, SOTA-metaheuristic comparison, ZST: four of the five objectives
   have no results. What is the deliverable?"

## 6. The honest reframe (recommended) — and what to run

**Headline claim (achievable, literature-grounded, novel per the survey's own gap analysis):**
*Adversarial (ATLA) training of a SAC dispatcher on a real road network yields policies that are
substantially more robust to targeted network disruption than non-adversarially trained RL —
approaching the robustness of a fully-informed reactive dispatcher — and this robustness
transfers zero-shot.* Greedy becomes the reference line; the RL-vs-RL contrast carries the claim;
"where/when does adaptivity pay" curves (budget × load sweeps) become the secondary contribution;
the three near-washes become *motivating evidence* (§6 of the redesign said exactly this).

**The experiment that carries the thesis (Robustness Matrix):** on one fixed rung —
{**SACRED/ATLA**, **vanilla SAC** (no adversary), optionally **random-attack SAC** (domain
randomization control)} × attack portfolio {none, random blocks, scripted gateway-targeted,
learned best-response-per-policy} × ≥3 seeds, stochastic eval over paired instances, selection on
a validation attacker, reporting on held-out ones. Expected picture per RARL: vanilla collapses
under targeted attack, SACRED degrades gracefully at a small clean-performance premium. That
result is publishable-grade for an MSc *even if every policy loses to greedy*.

**Cheap rescue of sunk compute:** gen02 dynassign per-phase snapshots exist. Training only the
vanilla-SAC control (2–3 seeds × 800 ep ≈ overnight) and re-evaluating existing snapshots under a
portfolio turns Stage 1.5 from a wash into a robustness datapoint without retraining SACRED.

**If the hybrid rung is kept as the headline arena** (its strong-adversary property is exactly
what makes a robustness gap visible), it needs surgical fixes first — none of them big:
1. Cross-event claiming: exclude requests already assigned to another truck from
   `_assignment_candidates`; clear `assigned_target` if its node's demand hits zero (kills the
   zombie-orbit bug; ~5× faster episodes too).
2. Featurize `assigned_target` (own-target + other-target columns) and expose it in `observe()`.
3. Enable ETA/congestion-aware-distance features in static mode (parity of information with
   greedy; without this the anticipation story is unlearnable).
4. Reconsider γ (≈0.997–0.999 or per-decision discounting) so anticipation is inside the horizon.
5. Re-tune the adversary budget DOWN to the sweet spot via the (cheap) blockade-floor probe —
   target an attack that hurts (+30–50%) but does not saturate; and/or report the full budget
   sweep as curves.
6. Eval protocol: stochastic rollouts, attack portfolio, validation/test attacker split.

**Descope with the supervisor (explicitly, now):** SBO → future work; ERB-from-ALNS → one
ablation on the final rung; SOTA-metaheuristic baseline → rolling greedy-insertion is defensible
as "strong reactive dispatcher", a rolling-ALNS arm only if time allows; ZST → one held-out
demand-geometry (or second city) transfer of the final matrix policies — high value per CPU-hour.

**Decision points for Kilian:** (D1) adopt the robustness-headline reframe? (D2) arena = fixed
hybrid, or dynassign-rescue first, or both? (D3) which controls (vanilla only, or +random-attack)?
(D4) supervisor sign-off on the descope list. No training and no `src/` changes until these are
agreed.

---

*Artifacts: this file; `scratch/critique_probes.py` (+ raw output above). Suite state verified:
75 passed. CPU spent: test suite ~8 s + probes ~90 s. Nothing else touched.*
