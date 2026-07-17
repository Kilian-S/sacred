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
