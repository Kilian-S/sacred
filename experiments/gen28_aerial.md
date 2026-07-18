# Generation: gen28_aerial (the free-flight interdiction act: a continuous coverage boundary + the map-conditioning de-confounder)

- **status: PRE-REGISTERED 2026-07-16 night (AERIAL_BRANCH_HANDOFF.md; Kilian's decisions on
  record: TRAINED aerial result = MUST-HAVE; single UAV first; 2D sector, STATIC hazards;
  full autonomous build/screen authority, with Kilian's 2026-07-16 in-conversation amendment:
  PAUSE before any actual training run and present the screen verdict + compute envelope first).**
- **branch:** `gen28-aerial` (off `gen08-interdiction` @ `fe3a315`); all aerial code additive, new
  files only. **git SHA:** the commit landing this ledger + `src/envs/aerial_sector.py`.

## Why (the two things roads cannot give the thesis)

1. **A CONTINUOUS boundary axis.** On roads the boundary regime is K -> m with m a small integer
   (4-6), so the gen26 boundary map is a handful of discrete points. In open airspace the min-cut
   analogue is geometric: hazards of effective radius r in a sector of usable width W saturate the
   corridor as the coverage fraction **phi = 2Kr / W** grows. The boundary map becomes a smooth
   curve, with roads as the small-m corner of one game family.
2. **De-confounding map- from geometry-conditioning.** Every road threat map ever trained on is an
   affine transform of edge length (CRITIQUE_12-07-26 §3.1; zst_map_robustness). Aerial hazard
   layouts are placed INDEPENDENTLY of the lattice by construction, so a layout-generalist
   evaluated zero-shot on unseen layouts is the first genuinely map-conditioned transfer result
   available to the project.

## The game (pinned by the handoff; deviations recorded here)

- **Sector/lattice:** 2D rectangle, width W (lateral) x depth D. Waypoint grid ny x nx
  (lateral x depth), forward-progress DAG: from (col i, row j) to (i+1, j + dj), dj in {-1, 0, 1}
  (forward/diagonal only, no backward, no pure-lateral moves -> finite, enumerable route set).
  Base = (0, mid-row), target = (nx-1, mid-row). Unit spacing both axes; arc length 1 (forward)
  or sqrt(2) (diagonal); route travel cost = summed arc length.
- **Hazards:** each = centre c (from a CANDIDATE GRID of positions) + effective radius r.
  **Pinned proximity function (primary):** linear taper
  `p(arc) = p_max * max(0, 1 - dist(arc_midpoint, c) / r)`; **sensitivity row:** Gaussian
  `p_max * exp(-dist^2 / (2 sigma^2))`, sigma = r/2. Arcs outside every active radius are safe.
- **Interception semantics (recorded design decision):** every (arc, hazard) exposure is survived
  independently: `intercept(route, iset) = 1 - prod over arcs a of prod over hazards h in iset of
  (1 - p(a, h))`. This is the aerial `survival_intercept_fn` and keeps mission-failure a
  union-of-events coverage objective (the submodularity the greedy BR needs; NOT assumed, tested
  greedy-vs-exact at K <= 2 before any greedy-yardstick number is cited).
- **Interdictor:** commits K hazard centres from the candidate grid (hidden, pre-committed).
  Built AS an `InterdictionGame` (routes = lattice paths; interdiction sets = K-tuples of centre
  indices; payoff via the proximity survival product), so `solve()` (minimax LP),
  `best_response_attacker`, `cost_constrained_value` and `objective_matrix`/`solve_multiconvoy`
  (the fleet extension) apply verbatim.
- **Observable vs hidden split:** observable hazards (weather) are KNOWN to the defender and act
  as pure detour cost (an additive arc-cost penalty inside their radius, folded into
  `travel_cost`); hidden hazards (interdictors) are the strategic component. The defender should
  hedge only over the hidden part; no static lane rule does this by itself.
- **Objective:** single-UAV = interception probability (N=1 mission). Fleet extension (A4, later)
  = loss-averse mission P(>=1 lost), exactly as multiconvoy. Travel cost is a REPORTED premium
  column from the start.

## Baseline set (PRE-REGISTERED BEFORE THE SCREEN RUNS; the gen26/R0 dogma, applied prospectively)

Every screen cell and every future ladder in this act carries ALL of:

1. **shortest-path deterministic** (the operational default; scored worst-case);
2. **best deterministic route** (= loss_det, the certificate for the whole deterministic class);
3. **uniform-LANE stack**: `n_lanes = floor(W / 2r) + 1` maximally separated lateral lanes
   (clipped to the row count), played uniformly — THE disjoint-heuristic analogue;
4. **inverse-risk-weighted LANE stack**: lanes weighted by 1 / (max single-hazard exposure of the
   lane), the inv-vuln analogue;
5. **uniform-FULL-menu stack** and **inverse-risk-FULL-menu stack** (the gen26 second-pass
   lesson: at high coverage the strongest naive stack may live on the full menu, not the lanes);
6. **tabular smooth FP with the same BR oracle** (~20 lines, multiplicative weights, average
   strategy, drift-free): the gen26 lesson that "only self-play can train there" must be worded
   against best-response-oracle methods. This row is in the ladder BEFORE any deep-RL training,
   so the deep-RL act is framed honestly from the start: its unique claim is amortisation /
   generalisation across layouts (A3), never single-instance superiority over tabular FP.
7. **equilibrium (loss_mixed)** where the LP is exact; past the exact wall, the certified greedy
   yardstick `[v_greedy, v_greedy / (1 - 1/e)]` with measured low-K fidelity, as gen26.

**Aiming metric for the screen: LANE-heuristic/eq (and best-naive/eq = min of rows 3-5 over eq),
NOT det/eq** (the R0c dogma: det/eq measures where determinism fails, not where naive
randomisation fails). Non-degeneracy gates recorded per cell: eq value inside (0.02, 0.9);
defender leader entropy H/lnR materially < 1 (the F1 flat-landscape killer: a symmetric cell
with a uniform equilibrium is UNTRAINABLE by smooth FP and is excluded no matter how large its
heuristic gap).

## The screen (oracle-only, FREE, no training; `scratch/aerial_screen.py`)

Base geometry ny=9 x nx=13 (W=8, D=12), hazard candidate grid = interior columns {2,4,6,8,10} x
all 9 rows (45 positions), p_max=0.9, menu = all 9 lane paths + k-shortest padding + lateral-
diversity fill to ~R=60 (menu-sufficiency reported: eq value vs menu size). Sweep axes:

- **phi = 2Kr/W:** K in {1,2,3,4} (exact) x r in {0.8, 1.2, 1.6, 2.0} -> phi in [0.2, 2.0]
  (cells past phi ~ 1 document saturation); K=5+ via greedy yardstick only if needed;
- **taper:** linear (primary) vs Gaussian sigma=r/2 (sensitivity);
- **detour-cost weight:** lambda in {0, 0.01, 0.03} on normalised cost (the cost-weighted LP:
  defender minimises worst-case interception + lambda * expected cost);
- **hidden/observable mix:** 0, 1, 2 observable weather cells (fixed, asymmetric placement,
  cost-penalty within radius) alongside the hidden game;
- **pinch geometry:** an interior wall of blocked waypoints with a gap (lane counts vary along
  the path), on/off;
- **heterogeneous hazard effectiveness (added pre-run, the F1 lesson):** per-position p_max as a
  disclosed affine band of the centre's LATERAL position (`banded_pmax`, band (0.5, 0.95):
  terrain masking). A fully symmetric sector has a near-uniform lane equilibrium = the flat
  fictitious-play landscape that destabilised F1; this axis (plus pinch) supplies game-side
  asymmetry WITHOUT the road cost-vulnerability confound, and the screen records leader entropy
  per cell so only materially asymmetric cells reach the shortlist.

Deliverables: per-cell JSON (`models/runs/gen28_screen.json`) with all baseline rows + eq +
entropy + solve wall-times (the timing dogma: the oracle cost is measured BEFORE any training
envelope is projected); the phi-boundary figure (`assets/aerial_phi_boundary.png`); the A1/A2
instance shortlist = cells where best-naive/eq is materially > 1 (target the largest gap that
passes the non-degeneracy + asymmetry gates). If NO material gap exists anywhere, that is the
honest result and it goes to Kilian before any training CPU (the handoff's own instruction).

## The training acts (bars DRAFTED here; each act's final bar is pinned at ITS launch record,
## after the screen names the instance; NO training runs before Kilian's explicit go)

- **A1 (single-UAV feasibility slice), draft bar:** on the screen's headline cell, SACRED
  best-checkpoint TAP < the STRONGEST naive stack row (rows 3-5 above, same yardstick) on >= 2/3
  seeds AND pooled, AND within the certified band of loss_mixed; tabular-FP row reported beside
  it (not gated: it shares the oracle and is expected to reach ~eq; the deep-RL single-instance
  claim is feasibility + approach-to-equilibrium, never uniqueness).
- **A2 (the phi-boundary training curve), draft:** 3 seeds on the headline phi cell, 1 seed per
  remaining curve cell, all arms under one yardstick per cell; the trained curve overlays the
  oracle boundary figure (the act's product, the continuous analogue of
  `assets/k_boundary_map.png`).
- **A3 (the layout-generalist, the de-confounder), draft:** ONE policy trained across random
  HIDDEN-hazard layouts (geometry fixed, layouts re-sampled), zero-shot on unseen layouts vs each
  layout's own oracle rows (lane rows + random-init + eq). Primary: beats the strongest naive
  stack zero-shot on >= 4/6 held-out layouts, >= 2/3 seeds. This is the act the thesis's
  map-conditioning sentence has been missing; tabular FP CANNOT play here without solving each
  new layout at deployment (it is a solver, not a policy) — the honest uniqueness claim lives
  here, stated in exactly that form.
- **A4/A5 (fleet; aerial pattern-of-life):** recorded extensions, only after A1-A3; the dynamic
  register's naive baseline (anti-repeat over lanes) is pre-registered now for whenever A5 runs.

## RESULTS (appended per step; nothing above changes after the screen runs)

### SCREEN RESULT (2026-07-17, oracle-exact, 42 cells, seconds each;
### `scratch/aerial_screen.py` -> `models/runs/gen28_screen.json`;
### figure `assets/aerial_phi_boundary.png`; suite 179 green incl. 12 new aerial tests)

**Headline: the proximity mechanics deliver the gap the roads never had at K=1.** Unlike roads
(where the disjoint stack achieves the exact K=1 equilibrium), the best naive stack is 1.03-1.81x
the equilibrium on EVERY screened cell, at trainable leader entropy (H/lnR 0.19-0.64) everywhere:

1. **The driver is lane-count QUANTISATION, not phi alone** (the honest sharpening of the
   handoff's phi story): the gap is largest where the corridor width does not quotient into
   separated lanes (spacing ~ 2r: r=0.8 -> 1.57x, r=1.2 -> 1.45x) and smallest where it does
   (r=1.6: 3 lanes at spacing 4 = 2.5r -> 1.07-1.09x). Same-phi cells with different (K, r)
   differ (phi=0.4: K1r1.6 eq 0.300 vs K2r0.8 eq 0.163), so phi does not collapse the family;
   the figure plots per-r series and says so.
2. **Pinch cells are the strongest regime, by the predicted mechanism measured in full:** behind
   a 3-row wall gap, separated lanes cease to exist (the lane rule degenerates to the single
   surviving lane = the deterministic route, 0.771), the best naive becomes inv-risk-FULL
   (0.714), and the equilibrium (hedging over gap-crossing timing/approach) sits at 0.394:
   **best-naive/eq 1.81 (pinch+banded) / 1.72 (pinch)** at K=1, phi=0.40, H/lnR 0.19.
3. **Tabular smooth FP with the same BR oracle ties the equilibrium at every K<=3 cell**
   (e.g. 0.398 vs eq 0.394; 0.0828 vs 0.082), exactly as pre-registered: single-instance
   superiority is NOT on offer in this act either; the deep-RL claim is amortisation across
   layouts (A3), framed so from the start.
4. **Banded per-position effectiveness** shifts values but not the gap structure (1.07-1.52x);
   **Gaussian taper** agrees with linear (1.02-1.27x, same ordering) = the pinned-primary choice
   is not load-bearing. **Detour-cost weight is flat** (lambda 0.01/0.03 leaves the worst-case
   unchanged; equilibrium detour premium 14-19% vs the shortest route, reported as the cost
   column). **Timing:** exact solve <= 0.4 s at K<=3, 3-6 s at K=4 (149k isets); build < 0.1 s.

### A3 AIMING PROBE (2026-07-17, `scratch/aerial_layout_probe.py` ->
### `models/runs/gen28_layout_probe.json`)

12 spatially-correlated random effectiveness fields (RBF, rank-mapped into (0.30, 0.95),
independent of lattice geometry by construction), base sector, K=1, r=1.2:

| arm (per-layout, ratio to that layout's eq) | median | min | max |
|---|---|---|---|
| uniform-lane stack (layout-blind) | 1.55 | 1.47 | 1.99 |
| **inv-risk-lane stack (LAYOUT-AWARE two-line rule)** | **1.52** | 1.42 | 1.82 |
| inv-risk-full stack (layout-aware) | 2.89 | 2.55 | 3.25 |
| robust-static mixture (best single unconditioned object, fit IN-SAMPLE on all 12) | 1.14 | 1.03 | 1.64 |
| cross-play (another layout's equilibrium) | 1.42 (mean) | - | - |

Reading: the layout-aware naive rule stays ~1.5x suboptimal (A3 has a real, beatable target);
the robust-static cap is the honest strongest unconditioned object (in-sample-fit, disclosed:
fresh-layout performance would be worse) and up to 1.64x on hard layouts. **Menu sufficiency:**
eq stable R=20 -> 80 within 0.9% (pinch cell: identical to 4 d.p.), so R=40 menus are not an
artefact.

### ACT DESIGN DECISION (recorded 2026-07-17, before any trainer code): ONE generalist trainer
### carries all three acts

The single-instance single-UAV menu game is a single-state SAC bandit (the SYSTEM.md
saturating-bandit dogma: replay-state diversity is load-bearing; roads solved this with N=3
followers or walk-mode states, neither available at N=1), and the tabular-FP row makes
single-instance deep-RL superiority unclaimable anyway (screen row 3). So the aerial trainer is
built ONCE, as the gen15/16/27-pattern LAYOUT-GENERALIST: menu-select head, per-instance menus,
transferable per-route features ([normalised cost, layout exposure]), smooth-FP oracle-BR
attacker per instance, exact TAP estimator (the policy's route distribution under
`best_response_attacker`). The training POOL mixes instances (base + pinch lattices, banded +
random fields, K/r cells), so state diversity is structural and one run family yields: A1 = the
policy's rows ON the headline pinch cell (in-pool); A2 = its rows across the phi-grid cells
(the trained overlay of the boundary figure); A3 = zero-shot rows on HELD-OUT layouts (the
map-conditioning act). "Single UAV first" (Kilian's pin) = N=1 throughout; the fleet extension
(A4) stays recorded.

> **PRE-REGISTERED BARS (pinned now, before the trainer exists; training launch = Kilian's go):**
> - **A1 (headline cell pinch+banded K=1 r=1.6; anchors eq 0.394, best-naive 0.714, lane/det
>   0.771, tabular FP 0.398):** best-checkpoint TAP < 0.714 on >= 2/3 seeds AND pooled.
>   **STRONG:** pooled <= 0.55 (halfway naive -> eq). Tabular-FP row reported beside it, ungated.
> - **A3 (PRIMARY of the act; 6 held-out layouts, seeds 2000-2005, never trained):** zero-shot
>   mean ratio-to-eq < the inv-risk-lane rule's ratio on >= 4/6 held-out layouts AND pooled, on
>   >= 2/3 seeds, select-on-train. **STRONG:** pooled <= 1.25 (halfway inv-risk-lane -> eq).
>   **MECHANISM ROW (the de-confounder):** the same policy fed a PERMUTED field on the held-out
>   layouts must degrade materially toward the unconditioned caps (robust-static 1.14 in-sample /
>   cross-play 1.42); if it does not, the map-conditioning sentence is NOT earned and the act
>   re-scopes honestly (the zst_map_robustness lesson, applied prospectively).
> - **A2 (reported curve, ungated):** 1 seed per remaining phi-grid cell family in-pool; the
>   trained overlay on `assets/aerial_phi_boundary.png`.
> - **Fail branches, all writable:** A3 partial (train-layouts yes, held-out no) = the transfer
>   boundary measured; A3 fail = the aerial boundary map + screen stand as the act's product.

### TRAINER BUILD RECORD (2026-07-17; worktree `../sacred-aerial`, branch `gen28-aerial`;
### suite 183 green incl. 4 new contract tests)

- **`src/envs/aerial_interdiction_env.py`:** the thin adapter presenting the aerial game
  through the road env's observation/menu contract (sorted zero-padded node ids so the
  2026-07-09 ordering bug class cannot recur; edge col 4 = the layout's per-arc worst
  single-hazard probability, the recorded observable-projection decision; per-route head
  features = [minmax cost, minmax layout exposure]). `route_one`, `featurize_state`,
  `node_index_map`, the menu head and the full ProtagonistSAC update path run UNCHANGED
  (tests/test_aerial_env_contract.py: featurisation, menu-index/sorted-row agreement,
  obj_matrix == payoff at N=1, end-to-end replay+update, exact-distribution sanity).
- **`scripts/train_aerial_generalist.py`:** the gen15/16-recipe generalist, N=1. Pool (fixed
  across seeds): 18 layout instances (base sector, K=1, r=1.2, RBF fields seeds 1000-1017)
  + the 5 screened cells (pinch_banded_K1_r1.6 = the A1 headline, pinch_K1_r1.6, base_K1_r0.8,
  base_K1_r1.6 = the honest low-gap point, base_K2_r1.2); held-out = 6 layouts (seeds
  2000-2005, never trained). Per-instance smooth FP (tau 0.05, window 250), analytic reward,
  head-term lr 3e-2, ent-frac 0.5, alpha floor 0.20, select-on-train, per-eval ckpts.
- **Held-out reference rows (built with the pool, before any training):** eq 0.152-0.196;
  inv-risk-lane 0.213-0.287 (ratio-to-eq 1.38-1.69: the A3 comparator margins); det 0.47-0.63.
- **Timing probe (B9 gate, 80 sorties, NOT a training run):** pool build 0.5 s; ~0.25 s/sortie
  with updates at `--threads 3`; full eval sweep (29 instances, exact) ~1 s. **Envelope:
  12,000 sorties/seed ~ 50-60 min; 3 seeds at 3-parallel ~ 1-1.5 h wall.** Probe anchors:
  untrained policy = ratio ~2.77-2.79 on train AND held-out (the random-init reference row);
  mechanism signature already visible pre-scale (route_feat_w[exposure] training negative,
  alpha annealing off the 1.0 init).
- **LAUNCH GATE (standing):** the 3-seed batch + any >= 240-sortie smoke await Kilian's
  explicit go (his 2026-07-16 in-conversation amendment to the handoff's autonomy grant).

---

## GAME V2 (2026-07-17, PRE-TRAINING; Kilian's realism challenge, accepted): curved flight,
## line-integral exposure, dense adversary. The v1 sections above are the DESIGN HISTORY;
## every v1 anchor number is RETIRED (nothing had trained) and re-derived below.

**Why (Kilian, verbatim intent):** straight lattice polylines look like train routing, not
flight; the adversary's 45 unit-spaced positions under-serve a continuous placement choice.
Both criticisms are substantive: the v1 "lane-count quantisation" finding was partly a GRID
ARTEFACT (integer rows), and per-arc Bernoulli exposure depends on an arbitrary arc
discretisation. **The three v2 changes (src/envs/aerial_curves.py; suite 191 incl. 8 new):**

1. **Routes = curvature-bounded Catmull-Rom curves** through lateral-offset control points at
   depth stations (what a waypoint autopilot flies; bank limit kappa_max = 1.5, obstacle
   rejection generic point-in-rectangle -> terrain/building polygons are the recorded v3
   drop-in). Menu = lanes at CONTINUOUS offsets first (the naive rule can now always space
   optimally = a STRONGER baseline), then seeded diverse curves; still a finite family, so the
   LP / greedy BR / menu head machinery is untouched.
2. **Exposure = survival line integral:** hazard rate lambda(s) = kappa * max(0, 1 - d/r),
   kappa = -ln(1 - p_max)/r, so a straight dead-centre transit is intercepted with probability
   EXACTLY p_max (the calibration that preserves p_max's meaning; regression-tested at 1%).
   Removes the arc-discretisation dependence entirely.
3. **Adversary grid densified to step 0.5** (323 positions; K <= 2 exact; K = 3 on step 1.0,
   disclosed) with a **grid-convergence certificate**: at K=1 the game value moves 0.4034 ->
   0.4065 -> 0.4074 across steps 1.0/0.5/0.25 (converged, +-0.2%); at K=2 it moves 0.709 ->
   0.750 from 1.0 -> 0.5 (not yet fully converged; disclosed; all comparative rows are
   same-grid, same-yardstick).

### V2 SCREEN RESULT (2026-07-17, 30 cells, oracle-exact; `scratch/aerial_screen.py` ->
### `models/runs/gen28_screen.json` v2; figure `assets/aerial_phi_boundary.png` rebuilt)

The integral exposure is deadlier everywhere (sustained proximity accumulates hazard) and the
strengthened continuous-lane rule closes part of v1's gap, exactly as honesty predicted; what
survives is real: **best naive stack = 1.03-1.59x the equilibrium across all cells**, largest
at K=1, r=1.2 (4 lanes, spacing 2.67 vs 2r=2.4: forced grazing): base 1.59 / pinch+banded 1.58
/ banded 1.57, all at trainable entropy (H/lnR 0.53-0.61). The v1 headline cell
(pinch_banded_K1_r1.6) drops to 1.26: the re-screen re-aimed the act, which is what screens
are for. The gap now DECAYS with coverage (1.59 at phi=0.3 -> ~1.03-1.06 at phi >= 1),
opposite in shape to the roads' K -> m boundary: in the air, learning's edge lives at LOW
coverage where calibrated grazing-avoidance matters; at saturation everything dies together.
Tabular FP ties the equilibrium at every cell it runs (e.g. 0.3635 vs eq 0.362), as
pre-registered. **Menu-relative-equilibrium convention + sensitivity (recorded):** the curved
continuum is never exhausted by a finite menu; eq drifts -5.3% (pinch+banded) / -2.4% (base)
from R=40 to R=80. All arms (naive stacks, tabular FP, SACRED, eq) share the SAME R=40 menu,
so every comparison is same-game (the road k-menu convention, CRITIQUE_EXAMINER §4.5), and
the R-sensitivity row is disclosed beside any absolute value.

### V2 A3 AIMING (layout probe rerun; `models/runs/gen28_layout_probe.json`)

12 RBF layouts, base sector, K=1, r=1.2: **inv-risk-lane (layout-aware two-line rule) median
1.57x eq** (1.25-1.79); uniform-lane 1.69; robust-static unconditioned cap median **1.83x**
(1.04-3.02, in-sample-fit, disclosed); cross-play mean 1.90x. The map-conditioning premium is
LARGER in v2: the act's target strengthens.

> **RE-PINNED BARS (v2 anchors; bar STRUCTURE unchanged from the pre-registration above;
> still no training has run):**
> - **A1 headline cell = pinch_banded_K1_r1.2** (eq 0.362, strongest naive stack 0.572,
>   det 0.695, tabular FP 0.3635): best-checkpoint TAP < 0.572 on >= 2/3 seeds AND pooled;
>   **STRONG:** pooled <= 0.47 (halfway naive -> eq).
> - **A3 PRIMARY, sharpened pre-training (the baseline-completeness dogma):** zero-shot
>   held-out layouts (2000-2005) beat each layout's **STRONGEST naive stack** (min over
>   uniform/inv-risk x lane/full-menu, per layout: on v2 the full-menu stacks sometimes edge
>   the lane rules) on >= 4/6 AND pooled, >= 2/3 seeds, select-on-train; **STRONG:** pooled
>   ratio <= halfway from the per-layout best-naive mean to 1.0. Context row (disclosed): the
>   UNTRAINED network already scores ~1.66-1.70 on this pool (near the lane-rule band, 2/6
>   under best-naive at probe scale), so the pass requires genuine calibration, not mere
>   spreading. Mechanism row (permuted field must degrade toward the unconditioned caps:
>   robust-static 1.83 / cross-play 1.90) unchanged.
> - **A2 curve cells (reported, ungated):** base r1.2 K{1,2}, base r0.8/r1.6 K1,
>   banded r1.2 K1 + the pinch cells; overlay on the rebuilt boundary figure.
> - **Trainer pool updated accordingly** (6 cells + 18 layouts train / 6 layouts held out);
>   `scripts/train_aerial_generalist.py` at this commit.

**Staged v3 (recorded, NOT scheduled):** real-terrain sector (obstacle polygons from
buildings/terrain, wind as anisotropic cost/rate) on the same machinery; Kilian's decision
2026-07-17: symmetric-rectangle v2 first, real-land upgrade later if the calendar allows.

---

## GAME V2.1 (2026-07-17, PRE-TRAINING; Kilian's "spawncamping" observation): TERMINAL
## STANDOFF ZONES. The v2 anchors two sections up are retired in turn (still nothing trained).

**The observation (from the interactive view, K=2 cell):** the equilibrium attacker parks its
hazards at the route-convergence funnel by the target. **This is a structural degeneracy, not
just unfairness:** ALL routes must pass the terminals, so a terminal hazard covers every route
at once and routing skill is irrelevant there; the road game could never do this (edge-disjoint
routes leave by different roads). It explains the hot v2 K=2 values (0.75) and the K>=2 gap
compression. **Fix: no enemy emplacement within safe_r = 3.0 of base or target**
(`dense_hazard_grid(..., safe_r)`), which is also the realistic model: friendly-controlled
terminal airspace, secured delivery zone; the contested space is the corridor. The aerial
min-cut now lives in the corridor, as the road min-cut does.

### V2.1 SCREEN (30 cells re-run; `models/runs/gen28_screen.json`): the fair game, measured

- Equilibria drop everywhere (terminal camping WAS carrying much of the interception: base
  K1 r1.2 eq 0.407 -> 0.21; base K2 r1.2 0.75 -> 0.42): the game is winnable for a good
  defender now, exactly what the fix intends.
- **On the open symmetric sector with a UNIFORM field, the lane rule is now near-optimal
  (best-naive/eq 1.06-1.18)** — the honest headline of the fair game, stated first: with
  protected terminals and freely placeable lanes, naive independence hedging suffices on
  featureless ground. Calibration pays where STRUCTURE breaks the lanes: **pinch + banded
  1.24-1.28 (the A1 headline family), banded fields 1.15-1.24, K=2 pinch 1.28.**
- Grid convergence under standoff: K=1 eq 0.196/0.212/0.217 at steps 1.0/0.5/0.25 (0.5 is
  within 2.3% of 0.25); K=2 0.391 -> 0.424 (1.0 -> 0.5, disclosed as before).
- **The A3 family probe (`scratch/aerial_family_probe.py`, 8 random layouts x 6 families):**
  heterogeneous fields RESTORE the open-sector gap (the lane rule reweights fixed lanes but
  cannot MOVE them to thread cold regions): BASE r1.6 K1 is the strongest family —
  best-naive/eq median 1.34 (min 1.20), robust-static cap 1.32, cross-play 1.80 — while pinch
  families lose layout-dependence (cross 1.28-1.48: the gap funnels everything). **A3 layout
  family = BASE, r=1.6, K=1** (12-layout probe rerun at the family: inv-risk-lane 1.34
  [1.20, 1.51], robust-static 1.32, cross-play 1.78, eq mean 0.208; menu-sensitivity R40->80
  ~3-4%, same-menu convention standing).

> **RE-PINNED BARS (v2.1, final pre-training set; structure unchanged throughout):**
> - **A1 headline cell = pinch_banded_K1_r1.6** (eq 0.519, strongest naive 0.665, det 0.719,
>   tabular FP 0.522): best-checkpoint TAP < 0.665 on >= 2/3 seeds AND pooled; **STRONG:**
>   pooled <= 0.59.
> - **A3 (the act's primary):** held-out layouts 2000-2005 (BASE r1.6 K1 family), beat each
>   layout's strongest naive stack on >= 4/6 AND pooled, >= 2/3 seeds, select-on-train;
>   **STRONG:** pooled <= 1.17 (halfway best-naive median 1.34 -> 1.0). Untrained-context and
>   permuted-field mechanism rows unchanged.
> - **A2 curve cells:** per the updated CELLS list in the trainer (pinch_banded K1 r1.6 + K2
>   r1.2; banded K1 r1.6; base K1 r0.8 / K1 r1.2 / K2 r1.2).
> - **The honest concession, pre-written:** on open featureless ground with protected
>   terminals, lane hedging suffices (1.06-1.18); the trained claim lives on structured
>   sectors (pinches, heterogeneous fields) and on never-seen layouts. This is the aerial
>   analogue of the roads' K << m concession, stated on our terms.

---

## GAME V2.2 (2026-07-17, PRE-TRAINING; Kilian's directive: shift advantage to SACRED through
## honest structure, never firepower). FINAL pre-training design; v2.1 bars superseded in turn.

**The measured answer to "more/larger radii?" (structure probe + screen):** firepower COMPRESSES
the relative gap (base K2 1.06, K3 1.05-1.16: saturation kills everyone equally); STRUCTURE
widens it. Probes (`scratch/aerial_structure_probe.py`, `aerial_family_probe.py` + inline
family re-runs, all oracle-exact, committed):

1. **Staggered double pinch (wall at x=4 open top, wall at x=8 open bottom: a forced S-turn):
   NO lane curve exists at all** (the naive lane rule structurally dies; its rows become
   full-menu stacks). Fixed-instance gaps 1.38-1.56; random-layout family **1.55 median /
   1.39 min** under the complete naive set.
2. **Mixed threat radii** (30% r=2.0 sites, 70% r=0.8 teams; per-position `r` now supported):
   the single-seed 1.63 collapsed to 1.21-1.23 median once the naive rule was given BOTH lane
   spacings — the recursive baseline-completeness lesson, caught pre-pin. Recorded as a
   realism axis, NOT an aiming axis.
3. **Menu completeness (binding design change):** `build_curve_menu` now always carries every
   canonical lane spacing (0.8/1.2/1.6/2.0) and the candidate-search cap is raised for
   constrained sectors; `all_lane_sets` + min-over-spacings defines `best_naive` everywhere.
   Richer menus IMPROVED the equilibria (defender-side), further demotions accepted: the
   single-pinch cell's gap collapses to ~1.06 (kept as an honest curve point).

**The v2.2 pool (trainer at this commit; pool build 90 s, deterministic across seeds):**
18 train layouts = 9 open-sector r=1.6 + 9 double-pinch r=1.2 (fields RBF-seeded 1000+/1100+);
6 cells (headline `dblpinch_banded_K1_r1.2`: eq 0.379, best-naive 0.550, det 0.760); held-out =
3 open-sector (2000-2002: best-naive/eq 1.52/1.31/1.26) + 3 double-pinch (2100-2102: 1.65/
1.42/1.57), pooled mean 1.455. **Untrained-network context (disclosed, probe-measured): on
lane-less sectors a near-uniform init MATCHES the uniform-menu stack (headline 0.548 vs naive
0.550 at init; held-out 1.75, beats best-naive 1/6),** so beating the naive row alone is not
evidence of learning there; the headline bar is GAP CLOSURE.

> **FINAL PRE-REGISTERED BARS (v2.2; nothing has trained; these supersede all above):**
> - **A1 (headline cell dblpinch_banded_K1_r1.2):** best-checkpoint TAP closes >= 50% of the
>   best-naive -> equilibrium gap (**<= 0.465**) on >= 2/3 seeds AND pooled. **STRONG:** >= 75%
>   (**<= 0.422**). Untrained row (~0.548) and tabular-FP row (recomputed at results time under
>   these menus, eval-only) reported beside it.
> - **A3 (the act's primary; 6 held-out layouts spanning BOTH families):** select-on-train TAP
>   beats each layout's best_naive on **>= 4/6 AND pooled, on >= 2/3 seeds** (untrained does
>   1/6 at 1.75 pooled). **STRONG:** pooled ratio-to-eq **<= 1.23** (halfway from the 1.455
>   best-naive mean). Mechanism row (permuted field) and worst-case-style honesty rows as
>   standing.
> - **A2:** the 6 cells reported as the trained boundary/structure curve, ungated.
> - **Claim shape if it lands:** *one policy, zero-shot on never-seen threat pictures across
>   BOTH an open sector and a structured corridor, beats the strongest naive rule family
>   (given every spacing) where fixed rules are 1.3-1.7x off optimal; on featureless ground
>   with secured terminals, simple lanes suffice and we say so.*

### LAUNCH RECORD (2026-07-17; Kilian's explicit go, with the system-load constraint)

- **Smoke (240 sorties, gate = plumbing + mechanism signature, NOT performance): PASSED** —
  end-to-end sound; route_feat_w trains to [-4.20, -3.13] (cost + exposure avoidance), alpha
  1.0 -> 0.41; ratios drift UP early (TRAIN 1.68 -> 1.93), the expected early-FP transient
  (24 instances share 240 sorties: near-empty attacker windows), disclosed here so the batch
  curves are read against it.
- **Batch:** 3 seeds x 12,000 sorties via `scratch/gen28_batch.sh` at SHA `65096a3`+launch
  commit; **thread discipline per Kilian's constraint** (past runs spiked ~40% system time):
  OMP/VECLIB/OPENBLAS/MKL = 1, torch intra-op = 2/seed, inter-op = 1, `nice -n 10`.
  **Measured at launch: 1.4% system time** (33% user, 65% idle during pool build).
  Outputs `models/runs/gen28_aerial/seed{0,1,2}.{json,log}` + per-eval checkpoints.
- Bars: the v2.2 block above, pinned pre-launch; select-on-train; TAP over last 3 evals;
  results appended below when the seeds land.

### RESULT (2026-07-17 evening, 3 seeds x 12,000 sorties, ~74 min at 3-parallel, system time
### low throughout): **BOTH PRIMARIES FAIL as pre-registered. Reported plainly.**

| seed | A1 headline best-ckpt TAP (bar <= 0.465) | A3 sel-on-train held-out pooled ratio | beats best-naive (bar >= 4/6) | sel-on-test (optimistic) |
|---|---|---|---|---|
| 0 | **0.456** @ 4000 (passes) | 1.70 | 2/6 | 1.60 |
| 1 | **0.464** @ 1000 (passes) | 1.77 | 2/6 | 1.70 |
| 2 | 0.478 @ 6000 (fails) | 1.84 | 2/6 | 1.71 |

> **A1 (gap closure >= 50% on the dblpinch cell): FAILS on the pooled clause by 0.001**
> (pooled best-ckpt 0.466 vs bar 0.465; 2/3 seeds pass individually). The bar is not moved;
> the margin is stated. Mid-run the cell genuinely improves 0.72 -> 0.46 (real learning
> signal, ~49.9% gap closure at best) before last-iterate drift takes it to 0.70-0.84.
> **A3 (zero-shot held-out layouts): FAILS decisively, 3/3 seeds** (beats 2/6 everywhere;
> pooled 1.77 +/- 0.06 ~ the untrained context row 1.75; vs-naive 1.20-1.30 > 1). STRONG
> bars: not reached. The pre-written fail branch stands: the aerial act's banked product so
> far is the SCREEN + structure/boundary story; no trained-positive sentence is licensed.

**Diagnosis (from the curves + telemetry, not vibes):** the policy never learned CALIBRATED
mixing; it learned cost-and-exposure AVOIDANCE and concentrated. The transferable head weights
ran to rw = [-5.4 .. -7.6, -0.7 .. -5.0] (a -7.6 cost weight is a near-argmax on the cheapest
routes: predictable, hence exploitable: exactly what the BR punishes zero-shot); alpha annealed
to its 0.20 floor everywhere; held-out ratios never dropped materially below the untrained
level. **The prime structural suspect is the pre-flagged saturating-bandit dogma (SYSTEM.md):
at N=1 with menu-select, each sortie contributes ONE state; 24 instances = 24 distinct
observations total; replay-state diversity, which the road record shows is load-bearing, is
absent.** Roads at N=1 passed only in WALK mode (multi-state next-hop, B2-P3); roads at
menu-select always had N=3 followers. The failed configuration is the one untested cell of
that grid (N=1 x menu-select), now measured.

**Pre-registered next step (ONE bounded structural re-aim, not a knob chase; launch = Kilian's
go):** rebuild the aerial policy as a NEXT-WAYPOINT WALKER on the DAG (the proven road
single-vehicle pattern): 12 decisions per sortie (12x replay diversity, position-conditioned
states), no menu head and no route-level head terms to railroad, exact exploitability via the
per-hazard survival DP over the policy's full path distribution (cheap: O(nodes x H)). Menu
baselines/eq stay as reference rows (the policy class strictly contains the menu class,
disclosed). Exit criterion, pre-committed: if the walker re-aim also fails its bars (one
attempt, same bars re-derived for the walker class), the aerial act closes as
screen + boundary product and the trained-positive burden stays on the road acts.

---

## V2.3 WALKER RE-AIM (2026-07-17 evening; Kilian's go: "build and then launch")

**Built (suite green; `src/envs/aerial_walker.py`, `scripts/train_aerial_walker.py`,
`tests/test_aerial_walker.py` incl. exact-DP-vs-brute-force):** node-mode walker on the
legal DAG (crash-proof pruning to target-reaching nodes funnels the final columns home);
GEOMETRY CLASS v2.3 = every arm scored as waypoint-leg polylines under the line integral
(dead-centre leg calibration exact; disclosed: spline rendering is cosmetic; the menu class is
a strict subset of the walker class, so menu-eq is a reference anchor, not the walker's
optimum). Reference menu = native DAG paths (lane paths per spacing where they exist - none on
the double pinch, correctly - + seeded diverse legal walks, R=40); complete naive family as
standing. Smooth-FP interdictor over trailing realised-survival windows (tau 0.05, window
250); exact walker exploitability by backward DP (K=2 via chunked pairs); expected path
length = the cost column.

**V2.3 anchors (refs probe `--sorties 0`, deterministic pool; geometry-class shift moves the
numbers vs v2.2, disclosed):** headline `dblpinch_banded_K1_r1.2` eq **0.468**, best-naive
**0.741** (naive/eq 1.58), det 0.792. Held-out best-naive/eq: B 1.37 / 1.03 / 1.24,
D 1.59 / 1.47 / 1.60 (mean 1.383). **Untrained context row: train 1.72, held-out 1.80,
beats-best-naive 0/6, headline 0.752 (WORSE than the naive row: no initialisation freebie in
walker mode).**

> **PRE-REGISTERED BARS (v2.3; structure identical to v2.2, anchors re-derived same-geometry):**
> - **A1 (headline cell):** best-checkpoint exact exploitability closes >= 50% of the
>   naive -> eq gap: **<= 0.605** on >= 2/3 seeds AND pooled. **STRONG:** >= 75% (**<= 0.536**).
> - **A3 (primary):** select-on-train, held-out layouts beat their best_naive on **>= 4/6 AND
>   pooled vs-naive < 1, on >= 2/3 seeds** (untrained: 0/6). **STRONG:** pooled ratio-to-eq
>   **<= 1.19** (halfway from the 1.383 best-naive mean).
> - Estimator: EXACT per-checkpoint exploitability (no TAP averaging needed: the walker policy
>   IS the mixed strategy; no Monte Carlo anywhere); best-checkpoint/select-on-train discipline
>   and disclosed drift as standing. Exit criterion (pre-committed above) unchanged: this is
>   the one bounded attempt.

### V2.3 RESULT (2026-07-17 night, 3 seeds x 12,000 sorties, ~89 min at 3-parallel, system
### time low): **BOTH BARS FAIL; THE PRE-COMMITTED EXIT CRITERION FIRES. THE ACT CLOSES.**

| seed | A1 headline best-ckpt (bar <= 0.605) | A3 held-out beats best-naive (bar >= 4/6) | pooled ratio-to-eq | pooled vs-naive |
|---|---|---|---|---|
| 0 | 0.717 | 1/6 | 1.84 | 1.37 |
| 1 | 0.706 | 0/6 | 1.80 | 1.33 |
| 2 | 0.723 | 0/6 | 1.81 | 1.34 |

**The failure signature differs from v2.2 and is diagnostic: the walker never learned AT ALL.**
Train-pool mean stays 1.71-2.15 across all 24 evals against an untrained 1.72; the headline
cell sits 0.72-0.77 throughout (untrained 0.752; the smoke's 240-sortie 0.689 did not
extrapolate); held-out equals untrained (1.80-1.84 vs 1.80). Per-family: the double-pinch
holdouts sit AT the naive rows (1.00-1.07x, as the diffuse near-uniform init already does);
the open-sector holdouts far above (1.44-2.00x). Candidate mechanisms, recorded not asserted:
terminal-only reward over 12-step episodes with ~500 sorties per instance is a far thinner
credit signal than the road walk-mode act that worked (B2-P3: one instance, 3,000+ sorties,
same machinery); the per-step entropy target (0.5 ln 3) compounds over 12 steps toward a
highly diffuse path distribution the alpha floor then defends. Establishing which would
require the single-instance trainability rung first (the road curriculum's own lesson:
competence precedes comparison), which is OUTSIDE this act's pre-committed budget.

**CLOSURE (per the exit criterion, verbatim from the pre-registration):** the aerial act
closes as the SCREEN + BOUNDARY product. **[SUPERSEDED 2026-07-17 night: Kilian REOPENED the
act with a new mandate and full launch authority ("make this work... until the aerial branch
provides a positive result"; walker judged a bad direction, back toward menu route choice).
The v3.0 record below is the new pre-registration; the v2.2/v2.3 negatives stand as measured
history.]** What it banks for the thesis: (i) the oracle-exact
screen arc (v1 -> v2.2: proximity exposure, standoff zones, complete naive families, the
structure-not-firepower finding, the grid-convergence certificate); (ii) the interactive
sector exhibit; (iii) a clean, pre-registered PAIR of trained negatives with distinct
mechanisms (menu-select N=1 = the saturating-bandit cell, measured; walker-generalist at this
budget = credit starvation under multi-instance terminal-reward FP), which extend the road
programme's "preconditions for adversarial training" chain into the aerial domain. The
trained-positive burden remains on the banked road acts (gen26 boundary map; gen27 dynamic
generalist). Reopening this act (e.g. single-instance walker rung first, larger per-instance
budgets, or w05-scale compute) is a NEW pre-registration and Kilian's explicit call; nothing
further trains under this ledger.

---

## V3.0 FLEET (2026-07-17 night; Kilian's reopening mandate + sign-offs: 3 drones, mission
## P(>=1 lost), Tier-2 zero-shot is the primary, M4 now / w05 later, FAR safe)

**Design = the road gen16 register transplanted verbatim onto the curved aerial game** (the
one configuration that both learned and transferred on roads): N=3 fleet-route menu-select
(three observations per sortie: the state diversity whose absence broke v2.2), loss-averse
mission objective (the B3 law: where randomisation is irreplaceable), per-instance smooth FP,
leader/follower entropy split (0.5/0.05, warmup 250), head features [cost, exposure] at lr
3e-2, select-on-train, stacked-occupancy TAP under the exact mission BR. Recorded deviations
from the road trainer: no stack-dup (uniform under fleet-route, disclosed); K=2 cell dropped
(exact fleet matrix ~4 GB; K axis returns via the greedy yardstick post-positive).

**Fleet screen + pool anchors (oracle-exact; `scratch/aerial_fleet_screen.py` + the pool
printout, deterministic):** structured (double-pinch) layouts carry the prize: gated held-out
best-naive/eq = 1.54 / 1.42 / 1.50 / 1.51 / 1.49 / 1.35 (mean 1.468; naive rows include stack
AND independent-mixing rules over every lane spacing and the full menu); open-sector layouts
1.10-1.20 (the standing concession, reported ungated); headline cell dblpinch_banded eq 0.538,
best-naive 0.754, det 0.923; tabular FP ties eq (0.555) as always. Smoke (240 sorties,
plumbing gate): healthy; beats-best-naive already 4/6 at sortie 240 (v2.2/v2.3 never exceeded
2/6 at any point); ~0.29 s/sortie.

> **PRE-REGISTERED BARS (v3.0; pinned before the batch):**
> - **TIER 2 = THE ACT'S PRIMARY (Kilian's preference): zero-shot on the 6 gated held-out
>   structured layouts (seeds 2100-2105): select-on-train stacked-TAP beats each layout's
>   best_naive on >= 4/6 AND pooled vs-naive < 1, on >= 2/3 seeds.** **STRONG:** pooled
>   ratio-to-eq <= 1.23 (halfway from the 1.468 mean). Open-sector holdouts (2000-2002)
>   reported beside, ungated.
> - **TIER 1 (headline cell dblpinch_banded_K1_r1.2):** best-checkpoint stacked-TAP < 0.754
>   (the strongest naive) on >= 2/3 seeds AND pooled; **STRONG:** <= 0.646 (50% gap closure
>   toward eq 0.538).
> - Estimator exact throughout (occupancy TAP over last 3 evals under the mission BR); best-
>   checkpoint/select-on-train discipline, drift disclosed; 3 seeds x 12,000 sorties; thread
>   caps + nice per the standing constraint. Iteration under the reopening mandate: further
>   attempts (if needed) are ledger amendments with disclosed changes, never silent re-rolls.

### V3.0 RESULT (2026-07-18 early, 3 seeds x 12,000 sorties, ~80 min): **TIER 1 PASSES
### (the act's FIRST positive pre-registered bar); TIER 2 FAILS 1/3. Reported plainly.**

| seed | Tier-1 headline best-ckpt (bar < 0.754) | Tier-2 gated beats (bar >= 4/6) | pooled vs-naive | ratio-to-eq |
|---|---|---|---|---|
| 0 | 0.790 (fail) | 1/6 | 1.12 | 1.64 |
| 1 | 0.739 (pass) | 1/6 (mid-run touched 5/6) | 1.08 | 1.58 |
| 2 | 0.710 (pass) | **4/6, vs-naive 0.99 (passes alone)** | 0.99 | 1.45 |

> **TIER 1 PRIMARY: PASS** (2/3 seeds + pooled 0.746 < 0.754). Thin (pooled margin ~1%; best
> seed closes 20% of the naive->eq gap); STRONG (<= 0.646) not met. **TIER 2 PRIMARY: FAIL**
> (1/3 seeds; bar needs 2/3). Iteration continues under the reopening mandate.

**Diagnosis (measured, third occurrence + a selection failure):** (i) the COST head input
railroaded again (rw[cost] -> -10..-12) despite the mission reward containing NO cost term:
a purely spurious channel at the head; (ii) train-mean checkpoint selection missed real
competence (seed 1 hit 5/6 gated beats + headline 0.74 mid-run; selection took a worse early
checkpoint because the train mean stays noisy-flat here, unlike the descending road curves).

### V3.1 AMENDMENT (pre-registered BEFORE launch; bars + gated test set UNCHANGED)

1. **Head features = [exposure] only** (the reward-irrelevant cost channel removed from the
   head; cost remains visible to the GNN via edge distances); head-term lr 3e-2 -> 1e-2.
2. **Checkpoint selection = VALIDATION mean ratio** over 4 fresh layouts (2 structured seeds
   3000-3001 + 2 open 3100-3101; never trained, never tested; the gen24 val-stop precedent).
   Select-on-train and select-on-test dual-reported as before.
3. Everything else byte-identical to v3.0 (pool, budgets, FP, bars, estimator).

### V3.1 RESULT (2026-07-18, 3 seeds x 12,000): **TIER 1 REPLICATES (3/3 seeds; now 5/6 seeds
### across two independent batches); TIER 2 fails at the frontier.**

| seed | Tier-1 headline best-ckpt (bar < 0.754) | Tier-2 gated beats @ val-selection | pooled vs-naive |
|---|---|---|---|
| 0 | 0.716 (pass) | 1/6 | 1.04 |
| 1 | 0.750 (pass) | 2/6 | 1.02 |
| 2 | 0.735 (pass) | 1/6 | 1.03 |

> **TIER 1: PASS 3/3 + pooled 0.734** (v3.0: 2/3 + 0.746) = a REPLICATED positive across
> batches (5/6 seeds total). STRONG (<= 0.646) unmet. Drift visibly cured post-amendment
> (headline stable 0.72-0.79 all run; no rw railroading, rw stays |w| < 0.2).
> **TIER 2: FAIL** (1-2/6), but the miss is now AT the naive frontier (pooled vs-naive
> 1.02-1.04; per-layout 0.93-1.09) vs v3.0's 1.08-1.12: zero-shot play matches the strongest
> naive rule and does not clear it. **Pattern note (binding for wording): this mirrors the
> ROAD generalist exactly (static zero-shot at-but-not-below its naive frontier; the
> disjoint-baseline finding), whose honest rescue was the DYNAMIC register (gen19/gen27),
> where every static rule is provably capped.**

### V3.2 AMENDMENT (the last STATIC push; disclosed; bars + gated test set unchanged):
24,000 sorties/seed (2x) + 36 train layouts (24 structured + 12 open, 2x variety), 3 seeds,
everything else v3.1-identical. In parallel, the DYNAMIC aerial act (the gen19/gen27
mechanism on the fleet game) is being built as the register where the zero-shot claim is
provable; its own pre-registration follows below when its yardsticks are computed.

---

## V4.0-DYN PRE-REGISTRATION (2026-07-18; the adaptive-enemy register on the aerial fleet
## game - the proven gen19/gen27 mechanism; screen `scratch/aerial_dyn_screen.py`)

**Game per layout:** fleet-route stacked (N=3, mission damage 1-(1-p)^3); enemy = softmax-BR
(tau=0.15) to the trailing **w=2** window of realised fleet routes; per-sortie ANALYTIC
expected damage; episodes = 40-sortie chains, gamma 0.95, window cleared per episode; window
route-frequency as a second head column beside exposure (head lr 3e-2, the gen19 value).

**Screen (held-out structured layouts 2100-2102, oracle/analytic):** iid_eq (static-equilibrium
play vs the adaptive enemy) 0.41-0.44; **best naive-DYNAMIC rule (rotation/anti-repeat over
every lane spacing + full menu) 0.47-0.56 = WORSE than static play at w=2** (naive avoidance
concentrates predictably); history_opt (exact RVI over the window MDP) **0.071-0.080**.
Operating point w=2 tau=0.15 (at tau=0.10 hist_opt degenerates toward 0.02; at w=3 rotation
partially works, 0.32-0.43, reported as the scope boundary). The winnable corridor below
EVERYTHING simple is ~0.33 wide - vs the static game's 2-4%.

> **PRE-REGISTERED BARS (v4.0-dyn; pinned before the trainer exists):** per gated held-out
> layout (2100-2105), rows computed at build: iid_eq; a MULTI-START LOCAL-SEARCH STATIC
> OPTIMUM (the gen27 amendment, up front this time); the full naive-dynamic family;
> history_opt (exact, w=2). **PRIMARY: the policy's EXACT stationary damage (window-chain
> power iteration, no Monte Carlo) < min(iid_eq, static local-opt, best naive-dynamic rule)
> on >= 4/6 gated layouts AND pooled, on >= 2/3 seeds, at the validation-selected
> checkpoint.** **STRONG: pooled <= 2.5x history_opt.** Open-sector context rows ungated.
> Estimator/selection/discipline as v3.1; worst-case-vs-committing-enemy row reported
> (the gen27 regime-conditionality sentence carries over verbatim).

### V3.2 RESULT (2026-07-18, 3 seeds x 24,000 sorties, 36 train layouts): **TIER 1 PASSES
### AGAIN (3/3; now 8/9 seeds across three batches, pooled 0.742/0.734/0.746). TIER 2 FAILS
### a third time AT the frontier (beats 2/1/1 of 6; vs-naive 1.01/1.02/1.05).**

**The static-register boundary is now measured across three disclosed amendments:** zero-shot
vs-naive converged 1.08-1.12 -> 1.02-1.04 -> 1.01-1.05 under 2x budget and 2x variety. The
recipe amortises TO the strongest naive frontier on unseen layouts and not below it - the
same asymptote the road generalist showed. **The static Tier-2 push CLOSES here (no further
static attempts); the banked static positives are Tier-1 (replicated, 8/9) and the frontier-
matching zero-shot row (vs-naive ~1.0, itself a claim: one policy re-derives the best naive
rule's performance on sight, without being told the rule). The zero-shot-SUPREMACY burden
moves to the dynamic register (v4.0-dyn, pre-registered above), as the road record predicted.**
