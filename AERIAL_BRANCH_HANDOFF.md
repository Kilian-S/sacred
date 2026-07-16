# AERIAL_BRANCH_HANDOFF.md: the free-flight interdiction act (build brief for a fresh Fable instance)

> **Provenance.** Written 2026-07-16 night by the Block-R instance, at Kilian's request, for a
> NEW Fable instance to execute autonomously. **Kilian's standing decisions (verbatim):** the
> TRAINED aerial result is a MUST-HAVE deliverable (not screen-and-defer); single UAV first,
> fleet later; 2D sector with STATIC hazards for the thesis version (drifting/anisotropic =
> recorded extension); timeline is not a constraint ("we have enough time, let's focus on
> building"). **You have full autonomous launch authority for this act** exactly as the Block-R
> session did: open the branch, build, run the free oracle screen, pre-register the ledger, and
> LAUNCH the training runs yourself — but keep the house discipline (pre-registered ledger with
> pinned SHA + bars BEFORE any CPU; suite green after any `src/` change with raw output pasted;
> numbers live in ledgers; report failures plainly; every fail branch is a writable boundary).
>
> **Assumed context.** You have read `HANDOVER.md` (top banner), `CRITIQUE_16-07-26.md` (the
> disjoint-baseline finding and WHY baseline-completeness is now pre-registered), and
> `experiments/gen26_kboundary.md` + `experiments/gen27_dynamic_generalist.md` (the two rescue
> acts this one is the third of). If you have not, read them first: this brief assumes you know
> what "the disjoint heuristic", "the greedy BR", "the boundary map", "menu-select", "TAP",
> "best-checkpoint discipline" and "smooth FP" mean, and you understand the project's central
> claim (calibrated randomised routing is unexploitable where deterministic routing is not).

---

## 0. Why this act exists (the two things roads cannot give the thesis)

1. **A CONTINUOUS boundary axis.** On roads the interesting regime is the interdiction budget K
   approaching the min-cut m, and m is a small integer (4-6), so the boundary map (gen26) is a
   handful of discrete points. In open airspace the min-cut analogue is GEOMETRIC: with hazards
   of effective radius r and a usable sector width W, the attacker saturates the corridor when
   roughly 2Kr approaches W. The boundary parameter becomes a CONTINUOUS coverage fraction
   **phi = 2Kr / W**, and the gen26 boundary-map story becomes a smooth curve with roads as the
   small-m corner of one family. This is the single strongest reason to build it: it unifies the
   two applications under one figure instead of two disjoint application chapters.
2. **It DE-CONFOUNDS map-conditioning from geometry-conditioning.** The sharpest unrepaired
   structural critique in the project (`CRITIQUE_12-07-26.md` §3.1, and `zst_map_robustness.md`'s
   finding that per-edge map-reading is NOT the transfer mechanism) is that every road threat map
   is an affine transform of edge length, so "conditions on the threat map" was never separable
   from "conditions on road geometry". In the air, hazard fields are placed INDEPENDENTLY of the
   lattice geometry BY CONSTRUCTION. A generalist trained across random hazard LAYOUTS and
   evaluated zero-shot on unseen layouts would be the first genuinely map-conditioned (not
   geometry-conditioned) transfer result in the project. Design the env so this experiment is
   reachable (hazards must be re-samplable independently of the lattice), even if you only run the
   single-instance act first.

---

## 1. The game (the design, pinned; deviate only with a recorded reason)

**Sector and lattice.** A 2D rectangular sector, width W (lateral) x depth D (base->target).
Discretise into a WAYPOINT LATTICE: a grid of waypoints with forward-progress edges (a DAG toward
the target; allow lateral and diagonal-forward moves, forbid backward moves so the route set is
finite and enumerable). Base at one edge midpoint, target at the opposite. The candidate route
menu = a screened family of ~50-200 lattice paths (k-shortest-by-length + a lateral-diversity
selection so the menu spans the corridor width, the aerial analogue of `build_route_set`'s
"disjoint paths + k-shortest"). **Single UAV first** (one vehicle per sortie; the N-convoy fleet
+ mission objective is the extension, added exactly as multiconvoy extended single-convoy).

**Hazards and proximity-graded risk (the core new mechanic).** Each hazard = a centre c in the
sector + an effective radius r. An arc (waypoint segment) that passes within r of an active
hazard centre carries a non-constant interception probability that is a FUNCTION OF PROXIMITY to
the centre, not a constant. **Pinned proximity function:** linear taper,
`p(arc) = p_max * max(0, 1 - dist(arc_midpoint, c) / r)` (so grazing the edge of a hazard is
cheap partial exposure, dead-centre is p_max), with a GAUSSIAN variant
`p_max * exp(-dist^2 / (2 sigma^2))` as the pre-registered sensitivity row. Arcs outside every
active hazard's radius are safe (p=0). This continuous risk is what makes aerial routes genuinely
DIFFERENT from one another (unlike roads, where the disjoint routes are near-equivalent) and is
the mechanism most likely to make calibrated mixing beat the lane heuristic.

**The interdictor** commits K hazard CENTRES from a candidate grid of positions (hidden,
pre-committed), maximising interception. This is the direct analogue of committing K edges;
the interdiction "set" is now K grid positions and the payoff is computed through the proximity
function. **Information split (pinned design choice, and a hypothesis for where the lane heuristic
dies):** distinguish OBSERVABLE hazards (severe weather cells: known to the defender, pure detour
COST, not part of the game) from HIDDEN hazards (interdictors/jammers: the strategic component the
defender must hedge over). The defender should randomise only over the HIDDEN component; no static
lane heuristic does this, so the hidden/observable mix is a prime axis for the screen.

**Objective.** Single-UAV: interception probability (the single-convoy analogue). Fleet extension:
loss-averse mission-failure P(>=1 lost), exactly as multiconvoy. Keep travel cost as a reported
premium column from the start (detour cost is load-bearing here in a way it was not on roads).

---

## 2. What reuses, what is new (the build is mostly composition)

**Reuses unchanged (do NOT reimplement):**
- The `InterdictionGame` abstraction (`src/baselines/interdiction_oracle.py`): routes x
  interdiction-sets payoff matrix, `solve()` (minimax LP), `best_response_attacker`,
  `cost_constrained_value`. Aerial routes are lattice paths; aerial interdiction sets are K-tuples
  of hazard centres; the payoff cell is `intercept_fn(route, hazard_set)` through the proximity
  function. Build the aerial game AS an `InterdictionGame` (or a thin subclass) so the LP,
  greedy-BR and menu-select machinery all apply verbatim.
- `greedy_br_attacker` + the env `greedy_br=True` mode (added gen26, `multiconvoy_oracle.py` +
  `multiconvoy_interdiction.py`): the matrix-free submodular BR with the (1-1/e) guarantee. VERIFY
  submodularity survives the proximity objective (mission-failure is still a union-of-events
  coverage over hazard-arc incidences, so it SHOULD; add a K<=2 greedy-vs-exact test like
  `tests/test_greedy_trainer.py`, do not assume). This is what lets the aerial boundary map extend
  past the exact wall into high phi.
- The SAC core, the menu-select route-index head, smooth-FP discipline (`fp_dynamics.py`),
  best-checkpoint TAP selection, per-eval checkpoints, the whole eval/pre-registration discipline.
  `scripts/train_multiconvoy.py` is the template trainer; the aerial trainer is a sibling.
- The generalist machinery (`scripts/train_generalist.py`) for the ZST-across-layouts extension.
- The dynamic-adversary machinery (`scripts/train_dyn_generalist.py`, gen27) if you do an aerial
  pattern-of-life act.

**New code (the only genuinely new pieces):**
- `src/envs/aerial_sector.py` (new): the lattice generator (waypoint grid + forward DAG + menu
  selection), the hazard model (centres, radii, the proximity `intercept_fn`, observable/hidden
  split), and an `AerialInterdictionEnv` that presents the SAME observation/menu interface the SAC
  head consumes (so `featurize_state` and the menu head work unchanged; you may need a small
  featuriser column for the observable-hazard field, use the width-slicing back-compat pattern).
- The LANE-HEURISTIC baselines (new, in the screen + a baselines module): `floor(W / 2r)` laterally
  separated lanes, played uniformly; an inverse-risk-weighted variant; an anti-repeat variant for
  any dynamic register. THESE ARE THE DISJOINT-HEURISTIC ANALOGUE — they MUST be in every ladder
  and in the screen's pre-registration (the gen26 lesson: the naive baseline goes in FIRST).
- Regression tests: lattice determinism, menu enumeration, proximity-function correctness, the
  env reproduces the oracle's loss_det/loss_mixed by Monte-Carlo (the G-M1-style fidelity gate),
  greedy-vs-exact at K<=2. Suite must stay green; paste raw output.

**Branch:** `git checkout -b gen28-aerial` off the current `gen08-interdiction` HEAD. Keep all
aerial code additive/flag-gated where it touches shared files (`interdiction_oracle.py`,
`multiconvoy_oracle.py`) so this branch does not disturb the road results; the env and trainer are
new files. Kilian is fine with codebase rewriting IF needed, but composition is preferred: try the
`InterdictionGame`-subclass route before any refactor.

---

## 3. THE SCREEN FIRST (free, oracle-only; this is the gen26 lesson applied prospectively)

**Do this BEFORE building the trainer.** Its job is NOT go/no-go (Kilian has committed to the
trained result) — its job is to AIM: find the sector geometry, hazard radius, budget and
information mix where calibrated mixing beats the lane heuristic by a MATERIAL factor, so the
trained run lands positive by construction. Same discipline that made gen26 pass on the first try.

`scratch/aerial_screen.py` (oracle-only, no training): sweep
- coverage fraction **phi = 2Kr/W** across [~0.15, ~0.9] (vary K, r, W);
- detour-cost weight (0 = pure game -> higher: does the cost-risk trade tilt the equilibrium off
  the lanes?);
- hidden/observable hazard mix (0 = all hidden -> higher observable fraction);
- proximity function (linear vs Gaussian);
- pinch-point geometry (constrictions that make lane counts vary along the path).

For each cell compute, EXACTLY (LP where feasible, greedy yardstick past the wall): equilibrium
`loss_mixed`, `loss_det`, and the LANE-HEURISTIC value (uniform + inverse-risk variants). Report
**heuristic/eq** as the aiming metric (NOT det/eq — the gen26 dogma). Deliverables: the
phi-boundary figure (the continuous analogue of `assets/k_boundary_map.png`), and the SHORTLIST of
instances where heuristic/eq is materially > 1 (target the band where it is largest but the game
is non-degenerate). Pre-register the screen's baseline set (lane variants) in the ledger before
running it. **Hypotheses to test (not conclusions):** the gap should be largest where (i)
proximity-graded risk + real detour cost make the cost-risk trade continuous, (ii) hidden and
observable hazards coexist (hedge only over the hidden part), (iii) pinch points make lane counts
vary. If the screen finds NO material gap anywhere (unlikely given the mechanics), that is itself
the honest result and you tell Kilian before spending training CPU — but the mechanics strongly
predict a gap, unlike the flat road-congestion landscape.

---

## 4. The training acts (pre-register each; launch autonomously per your authority)

Open `experiments/gen28_aerial.md` (pre-registration BEFORE CPU). Suggested arc, mirroring the
proven road sequence:

- **A1. Single-UAV feasibility slice** on the screen's headline instance: SACRED (smooth-FP vs the
  oracle/greedy BR) vs shortest-path vs lane-heuristic vs vanilla vs equilibrium. Primary:
  best-checkpoint TAP < the lane heuristic (the aerial disjoint-baseline bar) AND approaching
  loss_mixed. This is the aerial analogue of B2-P3 (single-convoy) — expect it to WORK where roads
  worked, because proximity-graded risk gives a genuinely non-uniform equilibrium (unlike the
  symmetric disjoint road instances that destabilised F1).
- **A2. The phi-boundary training curve:** train cells along the continuous coverage axis (low phi
  = lane heuristic suffices; mid phi = SACRED beats it; high phi past the exact wall = greedy
  yardstick, SACRED alone survives). This is the act's headline figure and the thesis's
  application-unifying result. 3 seeds on the headline phi cell, 1 elsewhere (the gen12/gen26
  curve discipline).
- **A3 (the de-confounder, high value):** the layout-generalist — train ONE policy across random
  HIDDEN-hazard layouts (geometry fixed, threat field re-sampled independently), evaluate zero-shot
  on unseen layouts against each layout's own oracle. THIS is the first genuinely map-conditioned
  (not geometry-conditioned) transfer result; it directly answers `zst_map_robustness.md`'s open
  finding. Include the lane-heuristic and a random-init reference.
- **A4 (optional, fleet):** N UAVs + mission objective, exactly as multiconvoy extended
  single-convoy; the correlation/stacking questions carry over. Only after A1-A3.
- **A5 (optional, dynamic):** an aerial pattern-of-life adversary (hazards re-placed against the
  UAV's recent lanes) — the gen27 mechanism in continuous space; a naive anti-repeat lane rule is
  the baseline. Only if the calendar allows; A1-A3 are the thesis core.

**Every ladder carries the lane-heuristic rows and a fleet-cost/detour column** (baseline
completeness is pre-registered, not added later — the whole reason this act exists is that we
learned that lesson expensively). **Best-checkpoint discipline + disclosed drift** as standing.

---

## 5. Risks and how the proven playbook handles them

- **"You shrank the problem until RL won."** Mitigation: the screen is oracle-only and
  pre-registered; the lane heuristic is in every ladder from the start; the phi-boundary curve
  shows the FULL range including where the heuristic wins (low phi). Same defence gen26 uses.
- **Symmetric/flat instability (the F1 killer).** Proximity-graded risk + detour cost should give
  a non-uniform equilibrium (the gradient smooth-FP needs), UNLIKE the symmetric disjoint road
  sweep. VERIFY at the oracle level in the screen (leader entropy H/lnR materially < 1) before
  training, exactly as the multiconvoy Fork-A screen did.
- **Menu enumeration blowup.** The lattice can have exponentially many paths; the MENU is a
  screened ~50-200 subset (k-shortest + lateral diversity). Report menu-sufficiency (does the
  equilibrium value stabilise as the menu grows?) as gen13/CRITIQUE_EXAMINER §5.2 did.
- **Greedy-BR validity under the proximity objective.** Do NOT assume submodularity; test
  greedy-vs-exact at K<=2 on a real aerial instance before citing any greedy-yardstick number.
- **Compute.** Single-UAV games are small; the lattice LP is the cost. Time the oracle in the
  screen before projecting the training envelope (the SYSTEM.md timing dogma). Cap ALL thread
  pools on multi-process launches (`OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1` + torch threads;
  the 2026-07-16 lesson).

---

## 6. Definition of done (what Kilian gets)

A `gen28-aerial` branch with: the env + oracle + lane baselines + tests (suite green); the
oracle-only screen + its phi-boundary figure; a pre-registered `experiments/gen28_aerial.md` with
the A1 feasibility slice PASSED (SACRED < lane heuristic, approaching equilibrium) and the
A2 phi-boundary training curve; ideally the A3 layout-generalist (the map-conditioning
de-confounder). Chronicle entry appended; HANDOVER banner refreshed; everything committed;
numbers only in the ledger. The thesis sentence this act earns: *the road and air applications
are one game family along a continuous coverage axis; calibrated adversarial routing beats the
geometric lane heuristic wherever the coverage fraction is non-trivial, on threat fields placed
independently of geometry — the first map-conditioned (not geometry-conditioned) transfer in the
project.*

*Every launch is yours to make under the granted authority; open the ledger before the CPU, and
tell Kilian when A1 lands (the first positive) and if the screen ever finds no gap (it should not).*
