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
