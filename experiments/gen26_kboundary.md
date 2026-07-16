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

### Step 1 RESULT (2026-07-16, 3 seeds, ~35 min at 3-parallel, SHA `c9c474a`): **PASSED every clause, STRONG met**

| seed | best-ckpt TAP @ sortie | best single-ckpt @ sortie | final TAP (drift, disclosed) |
|---|---|---|---|
| 0 | 0.656 @ 700 | 0.686 @ 600 | 0.795 |
| 1 | 0.647 @ 300 | 0.649 @ 500 | 0.954 |
| 2 | 0.690 @ 500 | 0.700 @ 200 | 0.833 |

> **Mean exact best-checkpoint TAP 0.664 +/- 0.018 (3 seeds): < the heuristic 0.738 on 3/3 seeds
> AND pooled (PRIMARY PASS); pooled 0.664 <= 0.68 (STRONG PASS).** Anchors: equilibrium 0.604,
> det/ALNS 0.933. Consistent with the gen12 single-seed 0.661 (different SHA, context only).
> The K=3 crossover is REAL at n=3: at K = m-1 (m=4 disjoint routes) trained calibration beats
> the strongest naive heuristic, closing 55% of the heuristic-to-equilibrium gap. Last-iterate
> drift persists as always (final TAP 0.80-0.95), best-checkpoint discipline standing.

### Step 2 RECORD (2026-07-16, SHA `77fe57f`): greedy-BR mode built, tested, smoked

Suite **167 passed** (161 + 6 new in `tests/test_greedy_trainer.py`: greedy == exact at K=1;
(1 - 1/e) bound at K=2; payoff-column agreement; flag-off path untouched). 60-sortie K=5 smoke
on 71-33 runs end-to-end matrix-free; its anchors reproduce the R0c screen exactly (uniform-
disjoint 0.705). **Fidelity on the step-3 instance (71-33, oracle-side, 5 random distributions
per K): greedy within 0.0% / 1.4% / 1.8% of the exact yardstick at K = 1 / 2 / 3** — far inside
the certified (1 - 1/e) worst case; reported wherever the greedy yardstick is cited.

### Step 3 AMENDMENT (2026-07-16, BEFORE launch, disclosed): the K=4 cell moves off 35-159

The step-3 cells above named 35-159 K=4 as the headline cell; the R0c boundary screen (run AFTER
this ledger was committed; artefact `models/runs/r0_screen.json`) shows 35-159 (m=4) SATURATES at
K=4/5: heuristic 0.966/0.985 ~ det 0.964/0.980 under the greedy yardstick — no defender has
meaningful room, so training there would spend CPU on a dead cell. Per the pre-registered
re-aim clause ("instances from R0c; prefer m=5-6"), the headline cell moves to the R0c-screened
**71-33 (m=6, R=11, E=43): K=5 (= m-1, 3 seeds)** with **K=6 (= m, 1 seed)** as the saturation
boundary point. Anchors under the common greedy yardstick (computed pre-launch): shortest-stack
0.833 > **uniform-disjoint-stack 0.705** > inv-vuln-disjoint-stack 0.638. The 35-159 K=4/5
saturation rows are reported as oracle-side boundary rows (no training). Wall statement
(honest): at K=5 the exact matrix on 71-33 is 286 x 962,598 (~2.2 GB, RAM-hostile at
3-parallel); at K=6 it is 286 x 6.1M (~14 GB, infeasible outright); labels do not exist at
either. **BARS UNCHANGED IN FORM: PRIMARY = SACRED best-ckpt TAP (greedy yardstick) < 0.705 on
>= 2/3 seeds AND pooled, on the 71-33 K=5 cell. STRONG: pooled < 0.638 (the inv-vuln variant).**

**Step 3 launch record:** config = the step-1 config with `--od 71-33 --K 5 --greedy-br`
(then K=6 seed 0); sorties 1200, eval-every 100, per-eval ckpts, seeds {0,1,2} 3-parallel via
`scratch/gen26_step3.sh`; SHA = the commit landing this amendment.

### Step 3 RESULT (2026-07-16, 71-33 m=6, greedy yardstick, SHA `152f880`): **PRIMARY PASSED at K=5; K=6 beats BOTH heuristic variants (prediction exceeded)**

**K=5 (= m-1, the headline cell, 3 seeds):**

| seed | best-ckpt TAP @ sortie | best single-ckpt @ sortie | final TAP |
|---|---|---|---|
| 0 | 0.690 @ 600 | 0.655 @ 600 | 0.708 |
| 1 | 0.656 @ 1200 | 0.653 @ 500 | 0.656 |
| 2 | 0.654 @ 900 | 0.662 @ 1200 | 0.659 |

> **Mean best-checkpoint TAP 0.667 +/- 0.016: < uniform-disjoint-stack 0.705 on 3/3 seeds AND
> pooled (PRIMARY PASS).** STRONG (< inv-vuln 0.638): NOT met, reported plainly — at K = m-1 the
> vulnerability-weighted heuristic remains ahead of the trained policy on this instance.

**K=6 (= m, the saturation boundary point, 1 seed):** best-ckpt TAP **0.718** @ 800 (final
0.769) vs inv-vuln-disjoint 0.766 < uniform-disjoint 0.800 < shortest-stack 0.833. **The
pre-registered expectation ("at K = m nobody wins") was WRONG in SACRED's favour: once the
interdiction budget covers the whole disjoint set, BOTH naive variants die and the trained
policy still finds shared-edge escapes** (single seed; a boundary point, not a headline).

**Side observation (chapter-worthy):** the last-iterate drift that plagues every K=1 result
nearly VANISHES at high K (K=5 finals 0.656-0.708 ~ bests; seed 1's final IS its best; K=1
finals were 0.80-0.95 off 0.25-0.28 bests). Plausible mechanism: at high K the attacker's
coverage pressure is strong everywhere on the simplex, so the uniform attractor that pulls the
last iterate off the K=1 hedge is much weaker. Best-checkpoint discipline retained regardless.

**The gen26 boundary map (the act's product; all same-yardstick, fidelity <= 1.8% at K <= 3):**

| K vs m | cell | uniform-disjoint | inv-vuln-disjoint | SACRED | read |
|---|---|---|---|---|---|
| K << m | 35-159 K=1 (m=4, exact) | 0.250 | 0.241 | 0.256 [0.246, 0.266] | heuristics suffice |
| K = m-1 | 35-159 K=3 (exact) | 0.738 | 0.737 | **0.664 +/- 0.018** | SACRED beats BOTH |
| K = m-1 | 71-33 K=5 (m=6, greedy, past the exact wall) | 0.705 | 0.638 | **0.667 +/- 0.016** | beats uniform; inv-vuln ahead |
| K = m | 71-33 K=6 (greedy, past the wall) | 0.800 | 0.766 | **0.718** (1 seed) | SACRED alone survives saturation |

**The claim gen26 banks (Obj-5, rescued):** *when the interdiction budget approaches and crosses
the min-cut — exactly the regime where the exact LP is infeasible (no labels: K=5 matrix 2.2 GB,
K=6 14 GB) and naive disjoint randomisation saturates — adversarially self-played SACRED beats
the strongest naive heuristics, with the certified-interval greedy yardstick and <= 1.8%
measured fidelity below the wall.* At K << m the heuristics suffice and the thesis says so.
