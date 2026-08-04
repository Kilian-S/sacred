# Generation: gen40_dyn_sensitivity (oracle-only structure sweep of the dynamic register)

- **status: PRE-REGISTERED 2026-08-04 (Kilian's in-conversation go for a comprehensive
  sensitivity sweep; oracle/eval-only, NO training anywhere).**
- **git SHA at registration: `96c327a` (clean tree).**

## Question

How does the structure of the pattern-of-life dynamic register move along four axes: the
adversary's memory window w, the disjoint-corridor count m, the interdiction budget K, and the
total menu size R (via the padding parameter k_extra)? Purpose: locate structurally where
learned history-aware play has headroom (SACRED's opportunity region) and where simple rules or
static play already collect everything. Kilian excluded solver/learner nuisance parameters; the
adversary temperature stays pinned at the standing operating point.

## What this probe can and cannot say (pre-committed reading rules)

1. This sweep produces NO new trained numbers. SACRED's realised performance exists at three
   banked anchors only: gen19 (m=4, w=3, K=1: 1.21x the exact optimum, rotation attains it),
   gen27 (m=3 pool, w=3, K=1: 0.639x the static cap, composed rule 0.50-0.61x, 1.97x the exact
   optimum), gen35 (m=6, w=3: tie with best rule at K=2, beats every rule at K=3, 1.38x the
   exact optimum). Nothing in this ledger may be quoted as a trained result.
2. The structural quantities read as follows. `best_rule / opt` is the slack no two-line rule
   collects (the opportunity for learned play); `iid_eq / opt` is the total value of history;
   `rotation / opt` diagnoses whether the deterministic schedule survives (the w-vs-m law).
3. Cross-m comparisons use ratios only; absolute values are instance-specific.

## Conventions (all exact; the corrected solvers)

- Adversary: softmax best response (counts-normalised, `softmax_br`) at tau = 0.15 to the
  trailing w-window of realised routes; N = 3 fleet, mission objective, band (0.15, 0.95),
  Kaliningrad graph (screened ODs; standing instances 35-159 and 71-33), `stacked_L`.
- Dynamic optimum: Karp minimum mean cycle (`scratch/dyn_exact.py::karp_mmc`), per binding
  rule 8. Where the full-menu state space R^w exceeds 3,200 states, the optimum is computed
  over the DISJOINT-CORE-restricted policy class (m^w states) and labelled `opt_core`;
  wherever both are computable the two are cross-checked and the discrepancy reported.
- Rules: best rotation over <= 20 seeded corridor orders; composed anti-repeat (uniform over
  core routes absent from the window; fallback uniform-core when all are punished), exact
  stationary value by damped power iteration; full-menu anti-repeat where R^w <= 3,200.
- Statics: equilibrium-mixture stationary value (`iid_eq`, exact where support^w <= 70,000,
  else marked absent), uniform-core, inverse-vulnerability-core, uniform-full-menu where
  feasible; `static_det` = best committed pure route.

## Grid

ODs: one screened Kaliningrad OD per m in {3, 5} (screen: seeded rng(0), deg >= 3 pairs,
base disjoint count = m, k8 menu R in [10, 14], one-shot value >= 0.05; Gdansk fallback for
m = 3 disclosed if the screen fails), plus the standing 35-159 (m = 4) and 71-33 (m = 6).
Crossed with k_extra in {0, 4, 8}, K in {1, 2, 3}, w in {1, 2, 3, 4, 5}. Any (od, K) whose
exact column count exceeds 250,000 interdiction sets is skipped and recorded.

Script: `scratch/gen40_dyn_sensitivity.py`; artefact `models/runs/gen40_dyn_sensitivity.json`.
Sanity anchors that must reproduce before results are read: 35-159 (K=1, w=3, kx=8)
iid_eq ~ 0.147, opt ~ 0.0413, rotation ~ 0.0413; 71-33 (kx=8, w=3) K=2 iid_eq ~ 0.1823 /
opt ~ 0.0657 / best rule ~ 0.0929, K=3 iid_eq ~ 0.2549 / opt ~ 0.1018 / best rule ~ 0.1539.

## RESULTS (2026-08-04, single run, 646 s, 180 cells, zero errors or skips)

Artefact `models/runs/gen40_dyn_sensitivity.json` (gitignored per standing convention;
regenerable exactly from the pinned script, which is fully seeded). Screened ODs:
m=3 Kaliningrad 23-242, m=5 Kaliningrad 29-80 (first screen hits, rng(0)); standing 35-159
(m=4) and 71-33 (m=6). **Every sanity anchor reproduced exactly** (35-159 K=1 w=3 kx=8:
opt 0.0413, rotation 0.0413, iid_eq 0.1468, static_det 0.613, v_eq 0.2061; 71-33 kx=8 w=3:
K=2 0.0657/0.0929/0.1823, K=3 0.1018/0.1539/0.2549). Core-restricted and full-menu optima
agree to < 0.005 on 127/135 dually-computable cells; the 8 exceptions are the padding
finding below.

### The window law (best two-line rule / exact optimum, k_extra=8)

| w | m=3 K1 | m=3 K2 | m=3 K3 | m=4 K1 | m=4 K2 | m=4 K3 | m=5 K1 | m=5 K2 | m=5 K3 | m=6 K1 | m=6 K2 | m=6 K3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1.10 | 1.16 | 1.15 | 1.05 | 1.15 | 1.18 | 1.38 | 1.89 | 1.87 | 1.92 | 3.30 | 3.31 |
| 2 | 1.00 | 1.00 | 1.00 | 1.04 | 1.04 | 1.05 | 1.30 | 1.43 | 1.54 | 1.43 | 1.93 | 2.14 |
| 3 | 1.78 | 1.53 | 1.38 | 1.00 | 1.00 | 1.00 | 1.05 | 1.14 | 1.23 | 1.24 | 1.41 | 1.51 |
| 4 | 1.60 | 1.54 | 1.37 | 1.36 | 1.20 | 1.16 | 1.02 | 1.01 | 1.00 | 1.20 | 1.27 | 1.33 |
| 5 | 1.13 | 1.07 | 1.08 | 1.67 | 1.44 | 1.31 | 1.34 | 1.28 | 1.22 | 1.25 | 1.30 | 1.34 |

Note the w=1 cells sit on tiny absolute optima at K=1 (0.004: a one-sortie memory is
trivially dodged by everything); the m>=5 K>=2 w<=2 cells are large-ratio AND
operationally-sized.

**Law 1 (the m-vs-w diagonal).** Two-line rules attain the exact optimum precisely at
w = m-1 (rotation always has exactly one unpunished corridor) and die at w in {m, m+1}
(rule/opt 1.4-1.9): the acute failure band. At w >= m+2 the window approximates the long-run
mixture, the enemy blurs toward the static responder, and rules partially recover
(m=3 w=5: 1.07-1.15).

**Law 2 (the many-corridor regime).** At m >= 5 with K >= 2 the optimal window schedule is a
non-trivial cycle no rotation or anti-repeat form matches, at EVERY w (m=6: 1.3-3.3x), and
the gap grows with both m and K (m=6 w=3: 1.24 -> 1.41 -> 1.51 across K). This is the gen35
regime; its trained crossing (K=3) sits exactly where this landscape says the slack is.

**Law 3 (value of history).** The static cap sits 2.2-3.7x above the exact optimum on every
w=3 cell of the grid (all m, all K): dynamism pays everywhere measured; who can collect it is
what Laws 1-2 partition.

### The menu-size axis (w=3; padding k_extra 0 -> 4 -> 8)

Padding barely moves the rules or the static cap, monotonically worsens the naive full-menu
anti-repeat trap (m=4 K=1: 0.045 -> 0.126 -> 0.203), and IMPROVES the true optimum where
w >= m and K >= 2: on m=3 the padded-menu optimum beats the core-restricted optimum by
14-17% (K=2: 0.1564 vs 0.1895; K=3: 0.2596 vs 0.3050 at kx=8; same pattern at kx=4;
smaller 6% effects at 35-159 K=3 w=1 and 29-80 K=3 w=3). **Padding is a trap for naive
dynamic play and a resource for optimal play, and the resource is invisible by construction
to any core-restricted rule (the composed rule included).** A menu-wide policy class is the
only measured object that can in principle collect it; whether trained SACRED does is NOT
established here (reading rule 1).

### Placement of the banked trained anchors on this landscape

- gen19 (m=4, K=1, w=3): a rule/opt = 1.00 cell; nothing to collect, SACRED could only tie
  (landed 1.21x). The instance choice, not the learner, capped that act.
- gen27 pool (m=3, K=1, w=3 structurally): the acute band (rule/opt 1.78-1.86 here;
  1.63-1.85 on the Gdansk pool). SACRED landed at 1.97x opt, slightly above the composed
  rule: opportunity real, partially collected.
- gen35 (m=6, w=3): rule/opt 1.41 (K=2) and 1.51 (K=3); SACRED tied the rule at K=2 and
  collected ~26% of the slack at K=3 (1.38x opt). Slack existing is necessary, not
  sufficient.

### Binding reading consequences

1. Any thesis sentence about "where learning pays in the dynamic register" may now cite the
   two-parameter law (w vs m, sharpened by K) instead of per-act anecdotes.
2. The padding finding licenses: "the composed rule is structurally capped away from the
   optimum on padded menus at w >= m, K >= 2 (14-17% on the m=3 instance)"; it does NOT
   license any claim that trained SACRED collects that margin.
3. w=1 ratios at K=1 must never be quoted without their absolute optima (degenerate cells).

## EXTENSION (pre-registered 2026-08-04, same session, Kilian's go: "each axis towards 10,
## not a new gen"). Registered BEFORE any extension CPU.

Same conventions as above; oracle-only; script `scratch/gen40_ext_sensitivity.py`; artefact
`models/runs/gen40_ext_sensitivity.json`. Reading rules 1-3 and the anchor discipline carry
over unchanged. Feasibility guards are pre-committed, and every guard-skipped cell is
recorded, never silently absent.

- **A, window to 10.** w in {6..10} x K in {1,2,3} at kx=8 on the four standing ODs. Exact
  optimum by Karp where the core state count m^w <= 8,000, else damped relative value
  iteration (memory-light; convergence flag recorded; tolerance 1e-9, iteration caps by
  size). Guards: m^w <= 2.0M and state-x-column work <= 6e9. Full-menu anti-repeat is not
  extended (R^w infeasible; characterised at w <= 3 above); rotation and composed anti-repeat
  always; statics under the same work guards.
- **B, corridors towards 10.** Screen Istanbul and Kyiv for ODs with base disjoint count in
  {7,8,9,10} (degree prefilter min(deg) >= m; <= 1,200 tries per city-m; menu screen relaxed
  to R <= m+9; one-shot value >= 0.05). Found ODs run at kx in {0,8}, K in {1,2,3},
  w in {1..5}. A dry screen is itself the reported result (real arterial networks may not
  offer such corridor counts).
- **C, exact budget to 5.** Raised column cap 1.0M (route-level payoff only; occupancy-matrix
  memory guard n_occ x n_isets <= 1.5e8). Feasible set measured from the run-1 artefact:
  K=4 exact at 23-242 kx=0, 35-159 kx=0, 71-33 kx in {0,4,8}; K=5 exact at 71-33 kx=0 only.
  w in {1..5} under the work guards. Everything else is past the exact wall and recorded so.
- **D, menu to ~20.** kx in {12, 16} at w=3, K in {1,2,3}, four standing ODs, same guards
  (the occupancy-matrix guard will gate K=3 at the largest menus; recorded).
- **E, the greedy-adversary budget ladder (A DIFFERENT GAME, binding rule 5: never mixed in
  one table with the softmax cells).** The enemy becomes the DETERMINISTIC greedy best
  response (`greedy_br_attacker`, the certified gen26 machinery) to the window's stack
  mixture, the tau -> 0 large-K analogue, computable at any K. kx=8, w=3, K in
  {1, 3, 4, 5, 6, 8, 10}, four ODs; optimum by Karp on the full-menu window graph; rotation,
  composed anti-repeat and the statics re-scored under the SAME greedy enemy. Sanity bar:
  at K=1 the greedy response must equal the argmax single edge exactly.

- **E REVISED (2026-08-04, same session, BEFORE tier E ran; the smoke is the reason and is
  recorded as a scoping fact).** The K=1 sanity bar PASSED on all 12 routes of 35-159, but
  the smoke's K=3 cell measured the deterministic-response game DEGENERATE: the exact
  optimum is 0.0000 (and plain rotation attains it), because under soft interception any
  route sharing no edge with the predicted set loses nothing, and a deterministic response
  is perfectly predictable from the window. This is the tau -> 0 degeneracy of the gen19
  sensitivity grid, reproduced at the set level; a deterministic extension adversary does
  not carry the game past the wall. REVISION: tier E's enemy becomes SOFTMAX (tau = 0.15,
  same semantics) over a per-state CANDIDATE POOL of interdiction sets, built
  deterministically per window-count signature: greedy sets against the window mixture,
  against each distinct windowed route's stack, against uniform-core and against the
  one-shot equilibrium mixture, plus the top-K single edges by window-weighted damage and
  by raw vulnerability (deduplicated; pool size <= ~8). K in {3, 4, 5, 6, 8, 10}, kx=8,
  w=3, four ODs. CALIBRATION, pre-committed: at every cell where the exact-softmax game
  exists (all four ODs at K=3; 71-33 at K=4 and 5 via tier C), the pool game's optimum,
  rules and cap are tabulated BESIDE the exact game's, and the distortion is reported
  before any K > 5 pool cell is interpreted. If the distortion is gross, the recorded
  conclusion is that no computable extension preserves the softmax game past the wall,
  and the wall binds the GAME, not merely the solver.

### EXTENSION RESULTS (2026-08-04; tier A 42 min, tiers B-E 38 min after a screen-loop
### repair; artefacts `models/runs/gen40_ext_sensitivity.json` + `gen40_ext_tierA.log`)

Process notes, disclosed. (i) The first launch hung in tier B: the screen demanded 1,200
distinct pairs from a 325-pair universe (Istanbul has 26 nodes of degree >= 7). Killed by
PID, loop rewritten to enumerate the bounded pair universe, relaunched tiers B-E; tier A's
completed cells are preserved in `gen40_ext_tierA.log` (4-decimal precision; the script
regenerates them deterministically via `--tiers A`... argv "A"). (ii) Large tier-A cells
carry the pre-registered UNCONV flag (RVI iteration cap before 1e-9); those cells are
estimates, marked * wherever quoted, and no conclusion below hinges on a starred cell.
(iii) Kyiv's arterial graph has maximum degree 6: no m >= 7 exists there at all.

**W, window to 10 (kx=8; best rule / optimum).** The w=5 "recovery" of the base grid was
divisibility structure, not a trend. Rules fail 1.2-2.1x at essentially every w >= m, with
the DEEPEST failures where w is a MULTIPLE of m (m=3: 1.78 at w=3, 2.10 at w=6, 1.95 at
w=9; m=4: 1.36 at w=4, 1.83 at w=8) and partial recoveries near w = 2m-1 (m=3 w=5: 1.13;
m=4 w=7: 1.18). Mechanism: at w = qm a rotation's window counts are perfectly balanced, the
enemy sees the uninformative uniform signature and aims at its unconditional best target,
and rotation walks into it; optimal play deliberately unbalances the window to steer the
aim. m=6 climbs monotonically to 1.74* at w=8 with no recovery in range. The w = m-1
rotation-optimality diagonal stays exact everywhere it appears.

**M, corridors to 9 (Istanbul screen: m=7 at 33-423, m=8 at 273-618, m=9 at 307-614; m=10
dry, and Kyiv dry at every target).** High-m cells at K <= 3 show modest rule failure
(w=3: 1.05-1.06 at m=7 K<=3; 1.22-1.25 at m=8; 1.33-1.36 at m=9), and the numbers line up
with the base grid ON THE COVERAGE FRACTION: m=9 K=3 (K/m=1/3) gives 1.36 ~ m=6 K=2 (1/3)
at 1.41. The corridor axis per se adds little at fixed K; K/m is the operative variable,
exactly the gen26 static law reappearing in the dynamic register.

**K, exact to 5.** On 71-33 (m=6, w=3) the slack keeps widening to the exact wall: 1.51
(K=3) -> 1.55 (K=4, all three menus) -> 1.56 (K=5, core menu), cap/opt ~2.0-2.25. On
35-159 (w=3 = m-1) rotation remains EXACTLY optimal at K=4 = m (rule/opt 1.00): the
w = m-1 diagonal is K-invariant even at full coverage. On 23-242 (m=3) K=4 > m compresses
the slack to 1.09 (absolute losses ~0.52: saturation).

**R, menus to ~20 (w=3).** Menu growth WIDENS the rule failure wherever slack exists
(m=3: 1.78 -> 1.91 -> 2.04 across kx 8/12/16; m=6: 1.51 -> 1.60) and never helps the rules;
the m=4 rotation diagonal is R-invariant (1.00 at R=4 through R=20, K=1-3). The padding
channel is a defender-side resource that grows with the menu.

**E, past the wall: THE PRE-COMMITTED NEGATIVE BRANCH FIRES.** The deterministic greedy
response was measured degenerate before the run (optimum exactly 0, smoke, recorded above).
The revised pool-softmax game's calibration against the exact game is GROSS AND ERRATIC:
at the overlap cells (all four ODs K=3; 71-33 K=4/5) the pool game reports rule/opt 4.95
where the exact game says 1.00 (35-159), 5.41 vs 1.23 (29-80), 5.42/4.08 vs 1.55/1.56
(71-33 K=4/5), agreeing only on 23-242 (1.35 vs 1.38). **Recorded conclusion, binding: no
computable extension tested preserves the softmax game past the enumeration wall; K beyond
~5 is a wall for the GAME, not merely for the solver, and no pool-ladder cell may be
quoted as evidence about the exact game.**

**Consolidated laws after the extension (supersede the base-grid wording where they
differ):**
1. The m-vs-w law, refined: rotation is exactly optimal at w = m-1 (K- and R-invariant);
   rules fail at essentially every other w >= m-ish, deepest at w a multiple of m.
2. The coverage law: rule failure at fixed w grows with K/m, not with m or K separately,
   and keeps growing to the exact wall (1.56 at K=5).
3. The menu law: padding widens the rules' failure and raises the optimal-play ceiling;
   it never helps a corridor-locked object.
4. The wall law: the exact game ends at K ~ 4-5 on these menus, and the tested extension
   families change the game rather than extend it.
