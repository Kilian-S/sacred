# REDESIGN_INTERDICTION.md: the winnable redesign (opened 2026-07-06)

> **STATUS: APPROVED DIRECTION (Kilian, 2026-07-06). GAME-THEORETICALLY PROVEN; build not yet
> started.** This is the project's CURRENT NORTH STAR: a new agent should read this file first
> (after `HANDOVER.md`). It supersedes the gen07 *destination-arena exploitability* attempt, which
> hit a structural wall (the flat attack landscape: `experiments/gen07_contested_matrix.md`,
> corrected-gate result). It keeps the SACRED invariants (SAC, protagonist/antagonist, RL) and
> Application 1 (contested autonomous resupply); it changes the PROBLEM so adversarial RL is
> *necessary* and *provably* beats the classical baseline. Proof: `scratch/interdiction_game_probe.py`.
>
> **Locked decisions (Kilian, 2026-07-06):** (1) adopt the interdiction-game redesign; (2) arena =
> the **Kaliningrad** graph (not a synthetic theatre), on a high-edge-connectivity OD pair;
> (3) **single convoy first** (nail the clean headline result before adding multi-convoy VRP
> richness); (4) freeze target unchanged (Aug 3); thesis due 28 Aug. Active plan: `ROADMAP.md`
> (interdiction build phases). Forward pre-registration: `experiments/gen08_interdiction.md`.

## 0. TL;DR

Every failure in the campaign traces to one root: the adversary was **congestion** -
*observable, reroutable, reversible*. Against that, a reactive dispatcher captures nearly all the
value (Ritzinger reactive-dominance), the attack landscape is flat (every block ≈ equally
damaging; the corrected BR gate proved a learned attacker cannot beat random), and anticipation
- the thing RL and adversarial training add: is worth ~0. **That is the wrong adversary.**

Application 1's real threat is **interdiction / ambush**: *hidden* (unseen until struck),
*irreversible* (interception, not delay), *pre-committed* (positioned before the sortie, against
your pattern). Against a hidden, irreversible, pre-committed threat, reactivity is useless and the
ONLY defence is **anticipation + unpredictable routing**: which is a **Stackelberg security game**
(the deployed ARMOR/PROTECT/AAMAS lineage). There, a deterministic router is maximally exploitable
and the minimax **mixed-strategy** router provably cuts interception. And the reversal that makes
this beautiful: **SAC's max-entropy objective, which sabotaged us in the queueing problem, is
exactly the mechanism that produces the mixed strategy the equilibrium demands.**

## 0.5 Why the pivot was necessary (the full evidence chain, for handover)

A new agent must understand this is not a whim: it is the forced conclusion of a long, rigorous,
pre-registered campaign. The chain:

1. **gen03-gen06 (the campaign, `SACRED_PROGRESS.md` 7-10):** adversarial co-training against a
   congestion adversary does NOT confer robustness and measurably WORSENS it (gen06:
   dD_targeted −881 ± 284). Robustness ranking greedy > vanilla > adversarially-trained. The
   learned adversary attacks *below random* (gen03/04); the reactive classical dispatcher is the
   most robust policy (Ritzinger reactive-dominance, measured in our own framework).
2. **The exploitability reframe (`DIRECTION.md`, 2026-07-06):** the right instinct: minimax
   training's native product is *worst-case* performance against a *strategic* adversary
   (exploitability), not average-case robustness. We built five learnability fixes (twin reward,
   entropy repair, curriculum, attacker population, credit horizon; all on branch
   `gen07-contested`, suite 109 green) and a pre-registered matrix.
3. **The arena-scoping probes (`experiments/gen07_contested_matrix.md`, `scratch/*probe*.py`):**
   before spending the matrix, we probed the lever. Findings: raising truck capacity DESTROYS the
   lever (it de-stresses the system); the load "sweet spot" was noise (powered sweep); crude
   unpredictability against the *reactive* congestion attacker is thin-to-negative.
4. **The corrected BR gate (the decisive test):** with ALL fixes applied (counterfactual reward +
   entropy repair + γ), a best-response attacker trained against greedy still lands at 0.35×
   random. Telemetry: entropy pinned ~2.2 while α collapsed to 0.08 → the near-uniform policy
   reflects **near-zero Q-spread**, not entropy pressure. **The unifying root cause:** on the
   stressed queueing arena the attack landscape is *flat* (every route-reach block causes similar
   cascading damage, so random ≈ optimal and there is no differentially-best attack to learn);
   where it might be differentiated (low stress) it is *thin*. **Flat-where-large,
   thin-where-differentiated**: a structural property of the "block edges in a queueing network"
   adversary, NOT a fixable optimisation issue. This explains gen03/04/06 mechanistically.
5. **The conclusion:** no reward/entropy/curriculum fix rescues a congestion adversary, because
   the failure is in the PROBLEM's game structure (observable + reroutable + reversible →
   reactive-dominated + flat). The only way to "make it work" is to change the adversary to one
   where anticipation and unpredictability are the *necessary* defence. That is interdiction.

The campaign was not wasted: it is the rigorous, published-shape negative that *motivates and
justifies* the redesign, and the evaluation methodology (pre-registration, competence gates,
held-out portfolios, gating expensive training on cheap probes) carries over intact.

## 1. The proof (game-theoretic, before any training)

`scratch/interdiction_game_probe.py` computes, on a network with base→FOB routes and an enemy that
commits K interdictors to edges:
- `loss_det` = interception probability of the best DETERMINISTIC route, worst-cased over the
  attacker's committed best response (= shortest-path / greedy / a collapsed vanilla-SAC policy).
- `loss_mixed` = the minimax value: the best MIXED router vs the best-responding interdictor
  (= what SACRED learns via SAC entropy + ATLA).

| instance | K | loss_det | loss_mixed | gap |
|---|---|---|---|---|
| synthetic corridor (3 disjoint routes) | 1 | 1.00 | 0.33 | **0.67** |
| Kaliningrad 110→135 (edge-conn 3) | 1 | 1.00 | 0.33 | **0.67** |
| Kaliningrad 33→71 (edge-conn 6) | 1 | 1.00 | **0.17** | **0.83** |
| Kaliningrad 33→71 | 3 | 1.00 | 0.50 | 0.50 |

A deterministic router is intercepted **100%** of the time; the mixed router **17-33%**. The gap
is the headline positive result, it holds on the REAL graph, and `loss_mixed` is a **computable
ground truth** to validate the learned policy against. The result is TUNABLE (K) with a wide
sweet spot, and it requires OD pairs with **edge-connectivity ≥ 3**: of which the graph has 190+
(realistic: you site logistics bases with multiple egress routes for exactly this resilience).

## 2. Why this structurally guarantees a positive finding (unlike the old design)

| property | old (congestion) | new (interdiction) |
|---|---|---|
| threat visibility | observable | **hidden** (unseen until struck) |
| reversibility | reroutable / delay | **irreversible** (interception) |
| attacker timing | continuous re-aim (reactive chaser) | **pre-committed** (Stackelberg) |
| reactive baseline | dominates (Ritzinger) | **useless** (ambush already set) |
| attack landscape | flat (every block ≈ equal) | **peaked** (chokepoints; disjoint routes matter) |
| defender's winning move | none over reactive greedy | **mixed strategy** (unpredictable routing) |
| SAC entropy | liability (wasteful randomness) | **the mechanism** (produces the mixed strategy) |
| ground truth | none | **minimax LP / double-oracle equilibrium** |
| adversarial training vs baseline | ≈ 0 (or worse) | **provably better** (det 100% → mixed 17-33%) |

The old design asked adversarial RL to beat a reactive dispatcher at a game reactivity wins. The
new design is a game reactivity *cannot* win, where the equilibrium *is* a mixed strategy and
adversarial minimax training is the canonical way to find it.

## 3. The concrete problem (build target)

**Adversarial convoy routing / resupply interdiction as a repeated Stackelberg game.**
- **Network:** the Kaliningrad graph (or a synthetic contested theatre); OD pairs (base→FOB) chosen
  with edge-connectivity ≥ 3. Multi-FOB / multi-convoy for VRP richness (Obj scope).
- **Protagonist (SAC dispatcher):** routes each convoy base→FOB. Stochastic policy = a mixed
  strategy over routes; its entropy is the unpredictability the equilibrium requires.
- **Antagonist (SAC interdictor):** commits K interdiction assets to edges each sortie, BEFORE the
  convoy's realised route, maximising expected interception. It best-responds to the defender's
  *policy* (its route distribution), which ATLA supplies (train interdictor vs frozen stochastic
  defender over many sorties → it learns where the defender tends to go).
- **Reward (high SNR, zero-sum):** protagonist = delivered value − interception loss − travel cost;
  antagonist = interception loss. Interception is a discrete, high-magnitude, directly-attributable
  event (struck on edge X because you took X and it was interdicted) → clean credit assignment,
  the SNR the old latency reward never had.
- **Dynamics (the "D" in SDVRP):** repeated sorties with evolving FOB demand and possibly network
  damage; the defender's mixed strategy and the attacker's allocation co-evolve (ATLA).
- **Optional richness:** value-differentiated cargo (blends Application 3 escort: protect
  high-value with more unpredictability); detection/survival probabilities (soft interception);
  partial network observability.

## 4. Research objectives (all get POSITIVE evidence)

1. **Obj 1 (zero-sum game):** the purest instance yet: a Stackelberg security game with a
   computable equilibrium; SACRED converges toward it (measurable).
2. **Obj 2 (sim env):** a targeted redesign of the game structure (information + reward + commit
   timing) on the EXISTING graph/GNN/SAC scaffolding: buildable, not from scratch.
3. **Obj 3 (SAC + ATLA + ERB):** ATLA = iterated best response ≈ fictitious play → equilibrium (its
   natural home); SAC entropy = the mixed strategy; ERB from the game-theoretic solver or the
   deterministic baseline. All finally *load-bearing and positive*.
4. **Obj 4 (SBO):** interdiction-aware base/FOB placement (site for egress connectivity): a
   natural, novel SBO instance.
5. **Obj 5 (eval vs baselines + non-adversarial SAC under disruption):** SACRED vs vanilla SAC vs
   shortest-path/greedy vs the **minimax equilibrium oracle** (computable on small instances). The
   headline positive: SACRED's interception under a best-response interdictor ≪ shortest-path's and
   vanilla's, approaching the equilibrium; exploitability gap large and validated.
6. **ZST:** the mixed-strategy concept transfers: train on one theatre, evaluate zero-shot on a
   held-out network / OD set.

## 5. Reuse vs new (buildable before the Aug 3 freeze)

- **Reuse:** the graph env + routing physics, GNN featurisation, SAC/ATLA machinery, the whole
  evaluation discipline (pre-registration, held-out portfolios, paired instances), the five gen07
  fixes where relevant.
- **New:** (a) the interdiction action + HIDDEN/committed information structure (the attacker's
  edge-selection action already resembles the congestion action: the change is timing +
  concealment); (b) the interception reward (discrete, attributable); (c) the equilibrium oracle
  (LP / double-oracle) for validation: already prototyped in the probe. This is a game-structure
  redesign on existing scaffolding, not a from-scratch env.

## 6. The pre-registered success criterion (draft)

On a fixed theatre + OD set + K sweep: **Expl(sacred) < Expl(vanilla) and < Expl(shortest-path)**,
where Expl = interception loss under a per-policy best-response interdictor over held-out sorties;
plus **sacred approaches loss_mixed** (the computed equilibrium) while shortest-path sits at
loss_det. Competence: sacred's clean (no-interdiction) delivery/cost within a small premium of
shortest-path. Full pre-registration in a new ledger (`gen08_interdiction`) before any training.

## 7. Honest risks + mitigations

- **"Shrunk until RL wins."** Mitigation: it's the canonical *deployed* security-game structure
  (Tambe et al.), the baseline (shortest-path) is the genuine operational default, and we validate
  against the true equilibrium. Choosing the problem where the thesis mechanism is the actual
  solution is good science, not gaming. Report the K sweep as curves, not a point.
- **Trivial/unwinnable regimes.** K < min-cut → mixing helps (our regime); K ≥ #disjoint routes →
  fully covered (report as the boundary). Tune K to the mixed-equilibrium band; wide sweet spot.
- **Learnability of the interdictor.** Now favourable: the reward is high-SNR and attributable, and
  the equilibrium is a genuine target (not a flat landscape). If the learned interdictor still
  lags, the equilibrium oracle provides the strong attacker for evaluation (unlike the old design,
  a strong attacker EXISTS and is computable).
- **Timeline.** Larger than an adaptation, but on existing scaffolding and with a structurally
  guaranteed result; gate the build on a cheap "does SACRED beat shortest-path on one theatre"
  slice before the full matrix.

## 8. Immediate next steps (build plan lives in `ROADMAP.md`)

1. ~~Kilian confirms the redesign direction.~~ **DONE 2026-07-06** (Kaliningrad, single convoy first).
2. Open `experiments/gen08_interdiction.md` (pre-registration): **stub committed; finalise before training.**
3. Build the interdiction env layer (hidden committed interdiction + interception reward) on the
   graph scaffolding; unit-test; wire `--problem interdiction`. **Single convoy, Kaliningrad,
   one high-edge-connectivity OD pair (e.g. 33→71, edge-conn 6).**
4. Cheap feasibility slice: train SACRED vs vanilla vs shortest-path on that one OD pair; show
   sacred's interception < baselines and → loss_mixed (the computed equilibrium). Gate the full
   matrix on it.
5. Full matrix + K / edge-connectivity sweeps + equilibrium validation + ZST; multi-convoy is a
   LATER richness add (single convoy first, per the locked decision).

## 9. Scope guardrails (single-convoy-first, Kilian 2026-07-06)

- **Single convoy** = one OD pair (base→FOB), one vehicle per sortie, choosing a route (a path or a
  distribution over paths). This is the cleanest instance of the security game and where the
  headline (det 100% → mixed 17-33%) lives. Do NOT add multi-convoy / multi-FOB coordination until
  the single-convoy positive result is banked.
- **Kaliningrad graph** (not synthetic): keeps continuity with the campaign and pre-empts the
  "toy" critique; the 190+ high-connectivity OD pairs make it a rich testbed. The synthetic
  corridor stays only as an illustrative sanity instance in the probe.
- **Route representation decision (for the build):** the policy can either emit a full path
  (sequence of next-hops, the existing next-hop routing machinery) or a distribution over a
  precomputed candidate route set. Start with next-hop routing (reuses `routing_mode`), so the
  mixed strategy emerges from SAC's per-step entropy; the candidate-set form is a fallback if
  credit assignment over long paths bites (the equilibrium oracle uses the candidate-set form).
