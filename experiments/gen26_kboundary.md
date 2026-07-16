# Generation: gen26_kboundary (Block R1: the K-to-min-cut act — where neither heuristics nor exact solvers can follow)

- **status: PRE-REGISTERED 2026-07-16 (Block R, `NEXT_STEPS_MASTER.md`; Kilian's full autonomous
  launch authority granted in-conversation 2026-07-16). Binding at launch; results appended per
  step.**
- **git SHA:** the commit landing this ledger (step 1); steps 2-3 pin their own SHAs at their
  launch records below.

## Why (CRITIQUE_16-07-26.md §1; the disjoint-baseline finding)

The two-line max-flow heuristic (uniform stack over the edge-disjoint routes) matches or beats
every trained static K=1 number. Its provable blind spot is BUDGET SATURATION: as K approaches
the number of disjoint routes m, the attacker can cover the disjoint set and the heuristic
degrades toward the deterministic value, while the equilibrium escapes through the shared-edge
menu. Measured seed of this act (single seed, gen12 cell, earlier SHA): at K=3 on 35-159 (m=4)
SACRED 0.661 < heuristic 0.738. K >= 4 is simultaneously (a) past the exact-LP RAM wall (no
labels, no exact yardstick), (b) past heuristic usefulness, (c) trainable only by self-play.
**The target claim: trained where neither exact solvers nor naive heuristics can follow.**

## Step 1: n=3 the K=3 crossover cell on 35-159 [PRE-REGISTERED HERE]

**Config (the gen13/gen12 lineage VERBATIM, K=3):** 35-159, k_extra 8 menu-select, band
0.15-0.95, N=3, K=3, fleet-route, smooth FP tau 0.05, switch-every 200, smooth-window 250,
leader-ent-frac 0.5, leader-alpha-floor 0.20, 1200 sorties, eval-every 100, EXACT estimator,
per-eval checkpoints, `--skip-vanilla` (the vanilla arm is not part of this bar; its hardcoded
fallback print stays out of citable output per the standing rule), seeds {0,1,2}, `--threads 3`
at 3-parallel. All three seeds run FRESH at this ledger's SHA (the gen12 0.661 was a different
SHA; never compare across git states — it is context, not a comparator).

**Oracle anchors (SHA-independent arithmetic, computed 2026-07-16,
`scratch/disjoint_baseline_probe.py`):** equilibrium 0.604; loss_det 0.933;
**uniform-disjoint-stack 0.738** (the comparator).

> **DECISION METRIC (PRIMARY): mean exact best-checkpoint TAP < 0.738 (the heuristic) on >= 2/3
> seeds AND pooled.** STRONG: pooled <= 0.68 (halfway to the equilibrium from the heuristic).
> FAIL: the K=3 crossover was seed noise; step 3 is re-aimed by the R0c screen (higher K and/or
> an m=5-6 instance) or the act falls back to the boundary-map framing (a writable, measured
> result either way). Best-checkpoint discipline and disclosed drift as standing.

**Command (pinned; per seed via `scratch/gen26_step1.sh`, detached, outputs
`models/runs/gen26_kboundary/`):**
```bash
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 35-159 --N 3 --K 3 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
  --skip-vanilla --seed $S --threads 3 \
  --json-out models/runs/gen26_kboundary/k3_seed$S.json \
  --ckpt-dir models/runs/gen26_kboundary/k3_seed${S}_ckpts
```

## Step 2 (build): the greedy-BR attacker in the trainer [design pre-registered; SHA at its record]

Wire the VERIFIED `greedy_br_attacker` (A4-core: exact at K <= 2, matrix-free to K = 5) into
`scripts/train_multiconvoy.py` behind a new flag (`--greedy-br`): (a) the smooth-FP attacker
refresh computes per-iset scores by greedy BR against the trailing-window occupancy support
instead of the exact matrix; (b) the exploitability eval scores occupancy distributions by
greedy BR; (c) the eager `obj_matrix` build is gated behind K <= 3 or the flag off. Regression
tests: greedy-vs-exact agreement in-trainer at K <= 2 on the live env; byte-identical behaviour
with the flag off. Suite green (raw output pasted at the record).

**Design decision (recorded now):** the smooth-FP softmax needs a per-iset score VECTOR, but
greedy BR returns one set. Resolution: at K >= 4 the attacker plays the greedy BR set with
probability (1 - eps) and, with eps = the standing fp-tau-derived exploration, a uniformly
sampled single-edge-perturbed variant (swap one edge for a random candidate edge) — preserving
the smooth-FP "punish the pattern, keep residual unpredictability" role without enumerating
C(E, K). The exact-mode path at K <= 3 is unchanged, so gen26 step 1 and every banked number
reproduce bit-for-bit with the flag off.

## Step 3: the K=4/K=5 cells [bars pre-registered; instances from R0c; SHA at its launch record]

Cells: 35-159 K=4 (3 seeds) and K=5 (1 seed) + one R0c-screened second OD (m = 5-6, K = m-1
and m; 1 seed each). Sorties 1200, eval-every 100, per-eval ckpts, greedy-BR mode.

**Yardstick:** ALL arms (deterministic/ALNS plan; uniform-disjoint-stack; inverse-vuln-disjoint
stack; SACRED best-checkpoint) scored under the SAME greedy BR; the certified interval
[v_greedy, v_greedy / (1 - 1/e)] reported for every absolute statement; same-yardstick
comparisons carry the claims.

> **DECISION METRIC (PRIMARY): SACRED best-ckpt (greedy yardstick) < uniform-disjoint-stack
> (same yardstick) on >= 2/3 seeds AND pooled, on the 35-159 K=4 cell.** STRONG: also < the
> inverse-vuln variant. FAIL branch (writable): "past saturation the learner no longer beats
> naive disjointness" = the measured upper edge of the boundary map.

## RESULTS (appended per step; nothing above changes after launch)
