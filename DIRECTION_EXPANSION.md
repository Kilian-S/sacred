# DIRECTION_EXPANSION.md: the post-gen13 computational expansion programme (opened 2026-07-10)

> **Provenance and status.** Direction set by Kilian 2026-07-10 (in-conversation, after gen13-lock):
> *"Disregard thesis writing, we have enough time for that."* The focus shifts to (1) more complex
> applications, (2) more comprehensive evidence on the five research objectives, (3) larger
> networks that prove ZST in a way an LP solve cannot match, and (4) SBO applied to logistically
> upstream decisions so the supply chain is treated holistically rather than as disjoint
> strategic/tactical/operational tiers. This file is the programme for the implementing instance:
> item list with justifications, codebase anchors, effort tags, and the recommended ordering at
> the end. Drafted by the critic instance (author of `CRITIQUE_PREFREEZE.md`) from the March 2026
> literature review's theoretical grounding plus a code-level capability audit.
>
> **Operating rules (unchanged, HARD):** never launch training without Kilian's explicit
> in-conversation go; every item gets its own pre-registered ledger (metric + gates + pinned SHA)
> BEFORE any CPU; numbers live only in ledgers; oracle/screen probes are free; plan first; no
> multiple-choice prompts to Kilian (prose + firm recommendation). Standing state to read first:
> `HANDOVER.md` top banner, `NIGHT_REPORT_2026-07-10.md`, `experiments/gen13_lock.md` (multi-convoy
> headline 0.274 +/- 0.025 post-fix on 35-159), `experiments/gen10_postfix.md` (single-convoy
> 0.276), `experiments/gen11_menuhead.md` (the B'/E' head-term mechanics and the lr-scale lesson),
> `CRITIQUE_PREFREEZE.md` (status banner = what is already executed).

---

## Axis A: ZST at a scale the LP cannot match

### A1. The generalist interdiction policy (ZST step 1: the keystone of the programme)

Train ONE policy across sampled OD pairs and vulnerability maps on Kaliningrad, conditioned on
the instance, and evaluate zero-shot on held-out ODs against each OD's own oracle equilibrium.

- **Justification (theory).** The review's justification section names ZST as the capability
  adversarial stress-testing should yield ("resilient, zero-shot transferable logistics policies
  that standard algorithms cannot achieve"); its catastrophic-overfitting citations (Kim et al.
  2020; Shaeiri et al. 2020; Jorge et al. 2022) hold that breadth of perturbation exposure is what
  prevents overfitting, i.e. multi-instance training is the theoretically indicated cure for the
  measured ZST-0 negative. ZST-0 (`experiments/zst_step0.md`) sharpened the hypothesis: transfer
  failed because nothing observable carried the threat map; give the policy the map and transfer
  has a mechanism.
- **Justification (codebase).** The conditioning mechanism exists and is proven trainable:
  gen11b's B' per-route cost/vulnerability head terms with `--head-term-lr 3e-2` reached O(1)
  weights with correctly-signed hedge behaviour. Add an edge-vulnerability observation column
  (width-slicing keeps back-compat), sample instances with the existing screen
  (`scratch/multiconvoy_instance_screen.py` / the F3 dataset generator), evaluate with
  `scratch/zst_transfer.py`.
- **What it buys.** The aim-level promise, trained; the enabler for A2, A3 and D3.
- **Effort:** ~2-4 days including a pre-registered gate (multi-instance training may need more
  sorties; treat sortie budget as the first probe).

### A2. Second-city zero-shot transfer

Train on Kaliningrad, evaluate zero-shot on a second OSM city (any geojson export; the loader
`src/utils/graph_utils.load_osm_graph_and_demands` is city-agnostic).

- **Justification.** The review's data-sources section explicitly promises OSM imports "to test
  the generalisation of the SACRED framework". OD-level transfer (A1) shares the map; city-level
  transfer is the graph-agnostic claim and pre-empts the single-map critique. The LP contrast is
  structural: an exact solver re-solves from scratch per city; the policy is a forward pass.
- **Effort:** ~1-2 days after A1 (data preparation + oracle screen + evaluation; no new training
  if the generalist transfers; one fine-tune arm as a disclosed fallback).

### A3. The amortisation benchmark (the honest successor to the retired wall-clock claim)

Stream M sampled instances (OD, threat map, N drawn fresh); measure cumulative compute and
time-to-first-decision for (a) LP re-solve per instance vs (b) one generalist forward pass per
instance; report the crossover in M and the latency distribution.

- **Justification.** The wall-clock scaling claim is dead (gen09 ledger supersession note: naive
  oracle N3K3 = 23 s, cheaper than training at every measured size). The defensible deployment
  claim is amortisation across a distribution of instances, and it deserves a measured curve
  rather than a sentence. Exact-methods infeasibility under real-time change is the review's core
  framing (Baldacci et al.), so this is the claim the literature actually supports.
- **Effort:** ~half a day of harness once A1 exists (the vectorised oracle and the exact policy
  evaluators already exist).

### A4. Matrix-free submodular greedy best response, then the large-K regime

Implement the greedy K-edge best-response attacker (for the mission objective, a defender
mixture's failure probability is monotone submodular in the interdicted edge set, so greedy
carries the (1 - 1/e) guarantee); verify against the exact BR at K <= 2; then train and evaluate
at K in {4, 5}, where C(79, K) is 1.5M-24M isets and the naive matrix is RAM-infeasible
(measured wall N5K3: 530 s, 4.8 GB).

- **Justification.** This is the point where the exact LP legitimately stops being the yardstick
  and bound-based evaluation takes over: the first regime where "the oracle cannot follow" is
  true rather than rhetorical. It also finally makes the implementation match the claim (the env
  currently materialises the objective matrix eagerly: `CRITIQUE_INTERDICTION.md` §5.4, still
  open). Design already specced in `CRITIQUE_INTERDICTION.md` §7 (scaling).
- **Effort:** ~1-2 days including verification; the K=4/5 training cells are then ordinary runs.

---

## Axis B: more complex applications

### B1. Dynamic multi-sortie campaigns (restore the S and the D of SDVRP)

Make the CAMPAIGN the episode: Poisson resupply demand arriving at FOBs, convoys scheduled across
successive sorties under fleet/turnaround constraints, and an interdictor that updates on the
defender's observed pattern of play within the episode (pattern-of-life inside one episode, not
only across FP iterations).

- **Justification (theory).** Answers the recorded problem-class critique (the headline game has
  no stochastic demand and no within-sortie dynamism; the S and D currently live in Acts I-II).
  Re-connects Ritzinger et al. 2015 on anticipatory vs reactive decisions, now inside a game
  where anticipation provably pays (the ambush is set against your pattern); connects the
  review's SDVRP taxonomy (Mardešić: dynamism entering through customer requests) to the
  security game.
- **Justification (codebase).** The Poisson demand engine, SMDP wrapper, latency accounting and
  per-request queues all exist from the campaign era (`graph_env.py`, `smdp_wrapper.py`); the
  interdiction layer already sits on the same GraphEnv. The build is composition, not greenfield.
- **What it buys.** The strongest "complex application" claim available; a genuine SDVRP-security
  hybrid no adjacent literature covers.
- **Effort:** ~1-2 weeks (the largest item; bound it with a pre-registered smoke gate and a kill
  date, since it can absorb unbounded time).

### B2. Learned-antagonist co-evolution (F2), with a PSRO-style population as the upgrade path

One post-fix attempt at a learned interdictor replacing the oracle BR on a headline instance
(evaluation stays oracle-BR portfolio-max regardless); if it bites, extend to a small
population/double-oracle loop (B4-full, already specced in `ROADMAP.md`).

- **Justification.** Obj-1's "environment-altering antagonist AGENT" has no learned agent in any
  positive result, and the pre-fix learned-antagonist evidence is confounded twice over (node
  ordering + the observation-staleness defect, `CRITIQUE_PREFREEZE.md` §5.2), so exactly one
  clean attempt is scientifically due. The review cites fictitious self-play (Heinrich and
  Silver) as the stabiliser, and a learned attacker is also the practical adversary in the
  large-K regime where even greedy oracle BRs get expensive (composes with A4).
- **Effort:** afternoon for the one-instance demo; ~2-3 days for a small population loop.

### B3. Heterogeneous fleet and value-differentiated cargo (the escort variant)

Convoys with unequal values (weighted mission objective); the equilibrium shifts randomisation
toward high-value assets; fleet composition becomes a real decision variable.

- **Justification.** Gives Obj-4's "fleet composition" content beyond N; realises the recorded
  Application-3 escort stretch (`DIRECTION.md` §1); feeds the D1 design space directly. Oracle
  extension is a weighted objective in `multiconvoy_oracle.objective_value` (small); the menu
  trainer needs a per-convoy value feature (the width-slicing pattern).
- **Effort:** ~2-3 days.

### B4. Correlated interception (one ambush team vs one stacked column)

Add a correlation parameter rho to the interception draws (independent -> comonotone) and sweep
it; report the headline ladder as a function of rho.

- **Justification.** Independence is the disclosed conservative assumption for stacking
  (`CRITIQUE_INTERDICTION.md` §3.3); the sweep converts a modelling caveat into an Obj-5
  robustness curve. Change confined to `caught_pmf` / the env's `resolve()`.
- **Effort:** ~half a day oracle-side plus one training curve.

### B5 (stretch). Deception: decoy convoys that shape the attacker's belief

A menu action that fields an empty decoy; the attacker's BR is computed against the OBSERVED
occupancy including decoys; the defender trades decoy cost against belief manipulation.

- **Justification.** Signalling/deception is where Stackelberg security games get theoretically
  deep, and it is genuinely novel for logistics interdiction. High risk (a new information
  structure, oracle no longer a plain matrix game).
- **Effort:** unbounded without discipline; attempt only after B1 or B2, with a hard gate.

---

## Axis C: more comprehensive evidence on the five objectives

### C1. Obj-3 ERB done literally: ALNS-demonstration bootstrapping

Seed the replay buffer with trajectories from the multi-convoy ALNS planner (and/or the
equilibrium mixture) and run a pre-registered time-to-competence ablation on 35-159.

- **Justification.** The review's Obj-3 wording promises "ERB bootstrapping via population-based
  metaheuristics" verbatim; gen01 left it inconclusive at n=1; the current realisation
  (demonstration bootstrapping) is honest but adapted. The machinery exists (`--leader-ckpt`
  forced-copy pattern, `--stack-dup`, the ALNS planner reaching loss_det). Closes the last
  adapted-wording gap in Obj-3.
- **Effort:** ~1 day.

### C2. Learned-follower coordination revisited, post-fix, on the favourable instance

Re-run the follow_w learned-coordination arc (the Obj-3 secondary) on 35-159 with the gen11b lr
recipe for all head-level parameters.

- **Justification.** The original arc was trained pre-fix on 62-97, the instance now known to be
  unfavourable, and its head-level parameters plausibly suffered the same silent lr no-op gen11
  diagnosed. If learned coordination beats structural stacking on the favourable instance, the
  single biggest remaining caveat on the multi-convoy act ("the stacking is structural, not
  learned") disappears.
- **Effort:** ~1-2 days.

### C3. Seed-strengthening plus the two missing headline rows (do first; hours)

Ten seeds on both headline cells (gen13 config on 35-159; gen10-SC config on 33-71) for real
confidence intervals; add the VANILLA row and the ALNS-forced-stack fairness row on 35-159 (the
named Obj-5 control and the fairness argument currently sit on 62-97 by analogy only).

- **Justification.** n=3 with population std is the recorded statistical weak point
  (`CRITIQUE_PREFREEZE.md` §4.3); runs are ~5 min/seed; the missing rows were flagged at the
  2026-07-10 re-sync and are trivial. Highest evidence-per-CPU-minute on the list.
- **Effort:** hours; fold into whichever batch runs first.

---

## Axis D: SBO as the holistic supply-chain layer

### D1. From surrogate regression to a genuine SBO loop over joint upstream design

Extend F3 into optimisation: design space = base/FOB placement x fleet size (x cargo mix once B3
exists); surrogate-guided acquisition (expected improvement or UCB over the MLP, or a small
ensemble for uncertainty) proposing designs; oracle evaluation of proposals; iterate; report
convergence vs random/grid search at matched evaluation budgets.

- **Justification (theory).** The review's SBO section is explicitly about metamodels permitting
  the COUPLING of canonical supply-chain problems that are traditionally solved disjointly
  (Blanning 1974; Sacks et al. 1989; Forrester and Keane 2009 on calibrating surrogate choice to
  budget; C. Wang et al. 2025 coupling two sequential EVRP decisions with a surrogate and showing
  cross-instance transfer). That is precisely the strategic/tactical/operational integration
  Kilian named as the direction. F3 proved the regression half (Spearman 0.894, argmin regret
  0.0); the acquisition loop is what makes it SBO rather than supervised learning.
- **Justification (codebase).** F3's dataset pipeline, features (including the closed-form
  harmonic-vulnerability aggregate) and by-placement validation split carry over;
  `src/sbo/flp_solver.py` provides the enumerate-and-argmin baseline to beat.
- **Effort:** ~2-3 days; oracle-only (no policy training), so probes are free.

### D2. Defender-side hardening: the tactical tier

Give the defender a pre-game budget to reduce edge vulnerabilities (escorts, route clearance);
optimise the allocation against the equilibrium (surrogate-guided or greedy with a submodularity
check); then train/evaluate the operational policy on the hardened network.

- **Justification.** Completes a three-tier computational stack on one game: harden the network
  (strategic investment), place bases and size the fleet (tactical, D1), randomised routing
  (operational, SACRED). The oracle already accepts arbitrary vulnerability maps
  (`length_band_vulnerability` is just a dict), so the build is thin; the interesting output is
  the interaction (does hardening change WHERE randomisation pays?).
- **Effort:** ~2-3 days.

### D3. The composite exhibit: surrogate over the TRAINED policy, priced by ZST

Fit the surrogate to (design -> the generalist policy's measured exploitability) rather than
(design -> oracle value), using the A1 generalist so each design costs one forward pass plus one
BR call instead of a retraining run; run the D1 acquisition loop on that target.

- **Justification.** This is where the three thesis pillars compose into one claim: the
  traditional objection to holistic supply-chain optimisation is that the operational layer is
  too expensive to evaluate inside a design loop; ZST makes it one forward pass; SBO makes it
  instant at query time; and no LP can participate at all (it would re-solve per design AND
  cannot evaluate a policy, only the equilibrium abstraction). The strongest candidate for the
  culminating computational exhibit of the whole project.
- **Effort:** ~1-2 days once A1 and D1 exist.

---

## Recommended ordering (firm; from the critic instance)

The dependency structure decides the order. A1 is the keystone (A2, A3 and D3 do not exist
without it); D1 is oracle-only and can run in parallel with anything; B1 is the largest item and
must not start until the keystone arc is banked.

1. **C3** first (hours): ten-seed the two headline cells; add the vanilla and forced-stack rows
   on 35-159. Closes the ladder while everything else is being planned.
2. **A1** (the keystone, ~2-4 days): pre-register the gate (held-out-OD exploitability vs each
   OD's oracle; a fine-tune arm as disclosed fallback), then train the generalist.
3. **D1** in parallel with A1's training runs (oracle-only, CPU-light): the SBO acquisition loop.
4. **A3** then **A2** (cheap harvests of A1): the amortisation benchmark, then the second city.
5. **D3** (the composite exhibit) once A1 and D1 are both banked.
6. **A4** (matrix-free greedy BR) then the **large-K cells**: the regime the LP cannot follow.
7. **C2** then **C1** (the Obj-3 completions on the favourable instance).
8. **B2** (one clean F2 attempt; extend to the population only if it bites).
9. **B1** (the campaign build) only now, with a pre-registered smoke gate and a kill date: it is
   the best application claim on the list and also the only item that can absorb unbounded time.
10. **B3/B4** as cheap riders attached to whichever multi-convoy batches run anyway; **B5** stays
    a recorded stretch behind B1/B2.

Drop order if time compresses: B5, then B1, then B3/B4, then C1, then A2. The keystone arc
(C3 -> A1 -> D1 -> A3 -> D3) should survive any schedule: it is the direct computational
realisation of all four of Kilian's stated direction axes at once.

*Every launch remains Kilian's explicit go; each item opens its own `experiments/<gen>.md`
pre-registration before any CPU, per the house rules.*

---

## Second-reader addendum (2026-07-10, the gen10-13 implementing instance; Kilian asked for concurrence + gaps)

**Concurrence:** the programme and the keystone arc (C3 -> A1 -> D1 parallel -> A3 -> A2 -> D3)
are endorsed as-is. The following amendments are incorporated by reference into the ordering.

1. **D2 was missing from the recommended ordering.** Slot it as a rider on D1 (same machinery) or
   immediately after D3.
2. **B0 (NEW, hard prerequisite of B1): the observation-staleness fix** (CRITIQUE_PREFREEZE §5.2:
   `GraphEnv.observe()` shares `_obs_nodes`/`_obs_edges` by reference and they mutate in place).
   B1 reuses exactly the Poisson/SMDP machinery this defect lives in; building the flagship
   application on it would reproduce the campaign confound inside the new headline act. Deep-copy
   or snapshot the two sub-dicts at transition creation + a contract test (a buffered state must
   be insensitive to later env mutation). ~half a day; gate B1 on it.
3. **A1 pre-registration must fix two design decisions:** (a) the generalist uses TRANSFERABLE
   features only - `route_feats` yes, `route_bias` NO (gen11b: identity capacity works but is
   definitionally non-transferable; including it re-creates the memorisation crutch the
   node-ordering bug used to provide); (b) the attacker regime under instance sampling - smooth-FP
   windows are per-instance and fill slowly under sampling; RECOMMENDED = each instance's
   equilibrium attacker (stationary, precomputed by the screen, cheap at K=1), with per-instance
   FP as a disclosed alternative arm.
4. **A3 must be quality-adjusted:** report the (compute, exploitability) FRONTIER with the
   generalist's amortised training cost included, not compute alone - the LP is slower AND exact;
   the policy is fast AND ~1.3x eq; only the frontier framing survives an OR examiner.
5. **A4 factual nuance:** K=4's naive matrix (4.4 GB, ~13 min projected) is heavy-but-feasible on
   the M4; **K=5 (~70 GB) is the honest "LP cannot follow" wall** - stake the claim there.
6. **C2 promoted above A4** (removing the "structural, not learned" stacking caveat is worth more
   than the large-K regime, at similar cost and with materially improved odds post-fix + lr-fixed
   head terms + favourable instance).
7. **B2 rides in parallel with A1's training days** (an afternoon, independent code paths) instead
   of waiting at position 8.
8. **C4 (NEW, bounded): ONE last-iterate convergence attempt** on 35-159 (annealed fp-tau or
   optimistic weight updates; pre-registered hold-the-tail bar; 1-2 days; hard gate, no chase).
   Both headlines carry the "equilibrium is a transient, best-checkpoint-selected" caveat; this is
   the only item on the list that can upgrade BOTH from "transient" to "converged". The old "no
   more leader experimentation" rule was scoped to the pre-fix 62-97 stabilisation chase; this is
   a new question on a new instance under the expanded mandate, and needs Kilian's explicit go
   like everything else.
9. **Calendar honesty:** the full list is ~4-6 serial weeks against ~2.5 experimental weeks before
   the 30 July Final Activities Report. The realistic envelope is the keystone arc + C2/C3/C4 +
   B2 riders; B1 (+B0) is the item most likely to die on the calendar, and the drop order already
   handles that correctly.

10. **B1-lite rungs (NEW, Kilian-approved 2026-07-10): two independently-reportable dynamism rungs
    written into B1 as its pre-registered smoke gates**, each restoring a letter of "SDVRP" to the
    headline game at a fraction of B1's cost, each with a computable yardstick:
    - **B1-lite-1 (within-episode pattern-of-life; D across sorties):** an episode = S sorties; at
      sortie t the interdictor softmax-best-responds to the defender's REALISED routes in sorties
      1..t-1 (the sample path, not the policy). The defender's optimal play becomes
      history-dependent (its own realised history is state: routes it has been seen on are hot);
      latency-free version isolates the pure strategic-dynamism effect. Small S, K=1: exactly
      solvable by backward induction over occupancy histories (the oracle discipline survives).
      ~2-3 days.
    - **B1-lite-2 (en-route threat revelation; D within a sortie):** walk mode + PER-EDGE soft
      interception resolved sequentially and observably as the convoy moves; mid-route detours
      become genuine recourse. K=1: backward induction on the trie. ~2-3 days.
    - Full B1 (Poisson demand + fleet scheduling + within-episode adaptive interdictor) then adds
      stochasticity in DATA on top, and the latency-vs-predictability coupling (serving promptly
      creates the pattern the enemy learns; unpredictability priced in latency) becomes the
      flagship claim. If the calendar kills full B1, either lite rung alone lets the thesis say S
      and D returned to the headline game.

**Amended ordering (net):** C3 -> A1 (B2 riding in parallel) -> D1 (parallel, oracle-only) ->
A3 -> A2 -> D3 (D2 as its rider) -> C4 -> C2 -> A4/large-K -> C1 -> B0 -> B1-lite-1 ->
B1-lite-2 -> B1 -> B3/B4 riders -> B5 stretch. Drop order: B5, B1, B1-lite-2, B3/B4, C1, A4, A2 -
the keystone arc plus C2/C3/C4 + B1-lite-1 survives any schedule.
