# Generation: gen43_unified_kboundary (the consolidated Act-2 instrument: one instance, both registers, one K-axis)

**status: PRE-REGISTERED 2026-08-08.** Mandate: Kilian's in-conversation direction (2026-08-08,
"Consolidation Act" session): the thesis's Act 2 currently stitches two experiments on two
instances with mismatched K columns (gen26 static: 35-159 K=1/3 + 71-33 K=5/6; gen35 dynamic:
71-33 K=2/3). Replace the stitch with ONE instrument: the 71-33 instance, both adversary
registers, one K-ladder run to each register's own measured wall. Banked cells are reused where
the licence below holds; new cells fill the rest. Kilian launches the batch himself (standing
launch workflow, 2026-08-05); Claude prepares and evaluates.

**git SHA:** the commit landing this ledger. Trainers and env are UNTOUCHED by this act (no
`src/` or `scripts/` change; the batch uses banked flags only).

## Question

On one instance (Königsberg 71-33: m=6 disjoint corridors, R=11 menu at k_extra=8, N=3 fleet,
band 0.15-0.95, mission objective), how does the value of trained calibration move along the
interdiction-budget axis K in BOTH registers, against the committed (static) adversary and the
observant (pattern-of-life, w=3 tau=0.15) adversary, each scored against its own exact or
certified benchmark and the complete naive family, with each axis run to its measured end?

## The free-work findings this act stands on (probe `scratch/gen43_consolidation_probe.py`, artefact `models/runs/gen43_consolidation_probe.json`, 2026-08-08, oracle/eval-only)

1. **Exact side reaches K=4.** The exact payoff matrix (286 x 123,410) builds in ~0.2 s and the
   LP solves in ~13 s. Exact equilibria on 71-33: v* = 0.1276 (K=1), 0.2553 (K=2), 0.3829
   (K=3), 0.5106 (K=4). K=2/3 values independently confirm the gen35 one-shot v_eq rows.
2. **The greedy yardstick is exact on stacks below the wall.** Greedy-vs-exact fidelity on all
   four stack arms at K=1..4: 0.0000 at 4 decimals (strengthens the gen26 <= 1.8% record,
   which was measured on random distributions).
3. **The low-budget concession is exact.** At K=1 the inverse-vulnerability disjoint stack
   attains v* exactly (0.1276).
4. **The static axis has a measured right edge.** Best-mixed-over-det (greedy yardstick):
   0.746 (K=5), 0.829 (K=6), 0.912 (K=7), 0.975 (K=8), >= 1.006 (K=9), >= 1.023 (K=10).
   Randomisation's value over the best single committed route (det pinned at 0.8325 from K=5
   up) reaches zero between K=8 and K=9; from K=9 the deterministic route is the best playable
   object. K=9/10 are therefore ORACLE-ROW cells; training there is a dead spend by the same
   R0c logic that killed 35-159 K=4/5.
5. **Dynamic K=4 is cheap and K=5 does not exist here.** At K=4 the adversary-side overhead is
   seconds (L build 27 s once; softmax 0.6 ms/sortie; evals ~1.2 s each); the cell costs the
   same ~2-2.5 h as K=3. The exact dynamic game at K=5 exists only on the kx=0 core menu (a
   DIFFERENT game, never mixed into this table; binding rule 5), and no computable adversary
   extension preserves the game past the wall (gen40 tier E, the pre-committed negative). The
   dynamic axis on this instance ends at K=4 because the game's computability ends, and the
   ledgered wall law is the closing sentence, not an omission.

## Reuse licence (the equivalence record, established 2026-08-08 before this registration)

Reused verbatim, with their original ledgers remaining the source of record:

| cell | source | SHA | pooled |
|---|---|---|---|
| static K=5 (3 seeds) | gen26 step 3 | `152f880` | 0.667 +/- 0.016 |
| static K=6 (3 seeds) | gen26 K=6 n=3 gate | `8ba949e` | 0.733 +/- 0.015 |
| dynamic K=2 (3 seeds) | gen35 | `5af4dd1` | 0.0934 |
| dynamic K=3 (3 seeds) + no-window control | gen35 | `5af4dd1` | 0.1406; control 0.2328 |

The licence rests on three verified facts. (i) `git diff <SHA>..HEAD` over both trainers and
`src/` shows exactly two additions since every reuse SHA (`networks.py` head_only branch,
`sac.py` _graph_key cache), both attribute/key-gated and unreachable by these trainers
(grep-verified; neither sets the gate). (ii) The game/oracle side reproduces byte-exactly at
HEAD (every gen26/gen35/gen40 anchor to the last printed decimal, probe above). (iii)
Bit-replay of the training loop is measured NOT to exist on this stack and never did: two
byte-identical uncapped invocations at HEAD differ from each other (0.8229 vs 0.8158 at sortie
100), a capped invocation differs from the banked gen35 trace by sub-1% trajectory noise, and
macOS updated 2026-07-20 between the two banked batches (venv unchanged since June). The
programme's reproducibility standard for trained numbers is pinned-SHA code identity plus n=3
seed spread, which every reused cell meets. The new K=5-adjacent cells double as a
distribution-level replication check (see bars).

## The game (pinned; both trainers verbatim, banked flags only)

- **Static register** (`scripts/train_multiconvoy.py`, the gen26 lineage): 71-33, N=3, k-extra
  8, menu-select, band 0.15,0.95, fleet-route, smooth FP tau 0.05, switch-every 200,
  smooth-window 250, leader-ent-frac 0.5, leader-alpha-floor 0.20, 1200 sorties, eval-every
  100, per-eval checkpoints, skip-vanilla; EXACT attacker at K <= 3, `--greedy-br` at K >= 4
  (fidelity finding 2). Seeds {0,1,2} 3-parallel, `--threads 3`, full thread caps.
- **Dynamic register** (`scripts/train_b1lite1.py`, gen19/gen35 mechanics UNMODIFIED): 71-33,
  w=3, tau=0.15, episode 40, gamma 0.95, 8000 sorties, eval-every 500, per-eval checkpoints.
  Seeds {0,1,2} 3-parallel, `--threads 3`, full thread caps. The trainer's internal
  `oracle_refs` history_opt is known-defective (undamped RVI) and is NOT citable; all cited
  dynamic refs come from the pinned probe JSONs (the gen35 rule, unchanged).

## Cells

- **Static NEW:** K in {1, 2, 3, 4, 7, 8} x 3 seeds. **REUSED:** K=5, K=6. **ORACLE-ONLY:**
  K=9, K=10 (finding 4; the death of mixing is the row).
- **Dynamic NEW:** K in {1, 4} x 3 seeds. **REUSED:** K=2, K=3 and the K=3 no-window causal
  control (the causal mechanism claim carries over; same trainer, same channel).
- **REPORTED rows (eval-only, post-batch):** tabular window-Q at matched budget at dynamic
  K=1 and K=4 (the gen35 machinery); worst-case committing row at K=4; tabular smooth FP
  (same greedy oracle) at every static K (probe already computed K >= 5; K <= 4 converges to
  v* by construction and is reported so).

## Baseline family (pre-registered, non-negotiable, all oracle/eval-only, per column)

- **Static, per K:** uniform-disjoint stack, inverse-vulnerability disjoint stack,
  uniform-full-menu stack, inverse-vulnerability-full-menu stack, static_det (best committed
  route), tabular smooth FP, exact v* at K <= 4; every arm scored under the column's single
  yardstick (exact at K <= 4, certified greedy at K >= 5 with finding 2 and the gen26 <= 1.8%
  record cited wherever greedy is the scorer). Pinned stack anchors (probe): K=1
  0.1666/0.1276/0.2252/0.2502; K=2 0.3288/0.2978/0.3812/0.3979; K=3 0.4675/0.4556/0.5014/
  0.5040; K=4 0.6017/0.5860/0.5899/0.5852; K=5..10 as in the probe artefact.
- **Dynamic, per K:** best-of-20-orders disjoint rotation, composed anti-repeat (core),
  full-menu anti-repeat, iid_eq mixture, static_det, exact dynamic optimum (Karp,
  `scratch/dyn_exact.py`). Pinned anchors (gen35 probe JSON + gen40 ext artefact, all exact):
  K=1 opt 0.0313 / best rule 0.0387 / iid_eq 0.0967; K=2 0.0657/0.0929/0.1823; K=3
  0.1018/0.1539/0.2549; K=4 0.1386/0.2152/0.3117.

## Decision metrics (PRE-REGISTERED; branches all writable)

> **DYNAMIC K=4 PRIMARY (the act's one live superiority bar):** best-checkpoint stationary
> per-sortie loss < the best naive rule 0.2152 on >= 2/3 seeds AND pooled (the gen35 K=3 bar
> form at the next budget). **STRONG:** pooled <= 1.15x the exact optimum (<= 0.1594).
> **FAIL branch (writable):** the beats-every-rule region is bounded at K=3 and the boundary
> is reported as measured, with the wall law closing the axis either way.
>
> **DYNAMIC K=1 (descriptive, expectation pre-written):** thin-slack cell (rules leave only
> 1.24x); expectation is a tie with or loss to the best rule, the low-K analogue of the
> static concession. SACRED beating every rule here would be a surprise reported as such.
>
> **STATIC K in {1,2,3,4} (descriptive, expectation pre-written):** the concession region.
> Expectation: SACRED does NOT beat the best stack at K in {1,2} (where stacks are exact or
> near-exact optimal) and narrows the gap with K; a crossing at K=4 would be an upgrade over
> the banked boundary map, reported plainly if it occurs. No superiority bar is set.
>
> **STATIC K in {7,8} (descriptive):** the frontier-collapse region. Expectation: SACRED
> within 2 pooled-sd of the best naive stack (tie-tracking), consistent with finding 4.
>
> **REPLICATION CLAUSE (the reuse check):** the new static K=4 and K=7 cells bracket the
> reused K=5/6; if the trained curve is discontinuous at the reuse seam (either reused cell
> sitting > 3 pooled-sd off the K=4-to-K=7 trend), the reused cells are re-run and the
> licence paragraph is amended. Best-checkpoint discipline and disclosed drift as standing;
> n=3 wording per binding rule 10.

## Design decisions ledgered

1. One instance carries both registers because m=4 instances have zero dynamic headroom
   (rotation exactly optimal at every K, gen40 law 1) and 35-159 saturates dead statically at
   K >= 4 (gen26 R0c). The banked 35-159 K=1/K=3 exact cells remain banked context for a
   one-sentence replication remark (the K = m-1 crossing on a second instance, exact
   yardstick); they do not enter the unified tables.
2. No new OD pairs are trained. Cross-instance generality of the LANDSCAPE is already
   oracle-exact in gen40 (the K/m coverage law across four Kaliningrad ODs and the Istanbul
   m=7-9 screens); cross-city generality of trained play is Act 3's flagship. This act's claim
   is per-instance shape, stated so.
3. Static K=9/10 are not trained (finding 4; the oracle row IS the result: the game ends).
4. Dynamic K=5+ is not attempted (gen40 wall law; the kx=0 K=5 game is a different menu and
   never mixes into this table).
5. Numbers live only in this ledger and its JSONs once results land; the thesis's Act-2 tables
   rebuild from here.

## Commands (pinned; KILIAN launches, per the standing workflow)

Batch `scratch/gen43_batch.sh` at this ledger's SHA: static cells K in {1,2,3,4,7,8} (each 3
seeds 3-parallel, exact attacker at K <= 3, greedy at K >= 4), then dynamic cells K in {1,4}
(each 3 seeds 3-parallel). Outputs under `models/runs/gen43_unified/`. Verification at
first-print level per the 2026-08-07 rule; `ps -o nice` check per the zsh nice-5 trap.

## Compute envelope (measured basis)

Static: ~12 min/seed solo at HEAD (200-sortie smoke pace, 1200 sorties), ~20-30 min per
3-seed cell at 3-parallel; six cells ~2-3 h. Dynamic: gen35 measured ~2-2.5 h per 3-seed cell;
K=1 lighter, K=4 the same plus ~30 s startup (finding 5); two cells ~4-5 h. Whole batch ~7-8 h,
one overnight run. RAM well inside envelope (largest object: three 11 x 123,410 float64 L
matrices, ~33 MB total; no eager obj_matrix at K >= 4 by the standing gate).

## RESULTS (appended per step; nothing above changes after launch)

### Batch RESULT (2026-08-08; launched by Kilian 11:42 BST at SHA `261703c`, ALL DONE 18:07
### BST, ~6.4 h; artefacts `models/runs/gen43_unified/` incl. per-eval checkpoints)

**Process disclosure (before any verdict).** The static K <= 3 cells trained to full
completion (1200 sorties, per-eval checkpoints saved, complete eval trace and final ladder
printed) and then CRASHED on the last line of the run while writing their JSON: the
exact-mode path serialises `classical_baselines`, whose `alns_plan` entry is a FleetPlan
object, and `float()` raised. A latent defect of the exact-path writer never exercised since
the greedy-BR era began (every post-`77fe57f` static run used `--greedy-br`, whose anchor
dict is float-only). No number is affected: the artefact of record for those three cells is
the `.log` (every eval printed) plus the checkpoint directories. The writer was repaired
AFTER results (filter to real numbers, this fold's commit), suite green pasted below;
training semantics untouched.

**Static register (best-checkpoint TAP, 3 seeds; anchors from the probe, exact yardstick at
K <= 4, certified greedy at K >= 5):**

| K | v* | best naive stack | SACRED per-seed | pooled | placement vs best stack |
|---|---|---|---|---|---|
| 1 | 0.1276 | inv-vuln-disjoint 0.1276 (= v*) | 0.163 / 0.162 / 0.156 | **0.160 +/- 0.003** | behind (nothing playable can beat v*); beats uniform-disjoint 0.1666 |
| 2 | 0.2553 | inv-vuln-disjoint 0.2978 | 0.325 / 0.335 / 0.324 | **0.328 +/- 0.005** | behind; ties uniform-disjoint 0.3288 |
| 3 | 0.3829 | inv-vuln-disjoint 0.4556 | 0.462 / 0.471 / 0.455 | **0.463 +/- 0.007** | behind by 0.007 (~1 sd), between the disjoint stacks |
| 4 | 0.5106 | inv-vuln-full 0.5852 | 0.579 / 0.614 / 0.622 | **0.605 +/- 0.018** | behind pooled; seed 0 below the stack (noted, no claim) |
| 5 | (wall) | inv-vuln-disjoint 0.638 | REUSED (gen26) | 0.667 +/- 0.016 | +0.029 above |
| 6 | (wall) | inv-vuln-full 0.730 | REUSED (gen26) | 0.733 +/- 0.015 | tie (+0.003) |
| 7 | (wall) | inv-vuln-full 0.7844 | 0.771 / 0.772 / 0.792 | **0.778 +/- 0.010** | -0.006 below on 2/3 seeds; within 1 pooled sd = a TIE by the standing n=3 wording (the gen26 K=6 standard); tabular FP ahead at 0.759 |
| 8 | (wall) | uniform-full 0.8216 | 0.820 / 0.819 / 0.826 | **0.822 +/- 0.003** | exact tie (+0.001); FP 0.812 |
| 9, 10 | (wall) | det 0.8325 OPTIMAL | ORACLE-ONLY | - | mixing dead (finding 4) |

**Every pre-registered static expectation held.** No crossing at K <= 4 (the concession
region confirmed, now with trained cells at every K); tie-tracking at K=7/8 (K=7 lands on
the favourable side of the tie band and is worded as a tie, not a win). **The replication
clause at the reuse seam is SATISFIED:** SACRED-minus-best-stack across K = 4..8 runs
+0.020, +0.029, +0.003, -0.006, +0.001 - a smooth monotone approach to the frontier with no
discontinuity at the reused K=5/6 cells, which therefore stand as licensed.

**Dynamic register (best-checkpoint stationary per-sortie loss, 3 seeds; exact anchors):**

| K | exact optimum | best rule | tabular window-Q (matched budget) | SACRED per-seed | pooled | verdict |
|---|---|---|---|---|---|---|
| 1 | 0.0313 | rotation 0.0387 | 0.0472 pooled (0.0504/0.0412/0.0500) | 0.0467 / 0.0468 / 0.0450 | **0.0462 +/- 0.0008** | rule ahead, as pre-written (thin-slack cell) |
| 2 | 0.0657 | 0.0929 | 0.1083 (gen35) | REUSED (gen35) | 0.0934 | tie at the rule |
| 3 | 0.1018 | 0.1539 | 0.1759 (gen35) | REUSED (gen35) | 0.1406 | beats every rule 3/3 (-8.6%) |
| 4 | 0.1386 | 0.2152 | 0.2169 pooled (0.1812/0.2253/0.2442) | 0.1774 / 0.1823 / 0.1863 | **0.1820 +/- 0.0036** | **PRIMARY PASS 3/3 AND pooled (-15.4%)** |

> **DYNAMIC K=4 VERDICT (per the pre-registered bars): PRIMARY PASS on every clause - all
> three seeds and the pooled mean sit 13-18% below the best naive rule.** STRONG FAIL,
> reported plainly: pooled 0.1820 = 1.313x the exact optimum (bar 1.15x) - the fourth
> consecutive dynamic cell where the trained policy beats the rules yet stays ~1.3-1.5x from
> the optimal cycle. Slack collected rises from ~26% (K=3) to **~43%** (K=4): the margin
> over every two-line rule WIDENS toward the wall, the gen40 coverage law now realised in
> trained play at every feasible K. The matched-budget tabular window-Q FAILS to beat the
> rule at both new cells (K=1 pooled 0.0472; K=4 pooled 0.2169 > rule 0.2152; one K=4 seed
> lands below the rule at 0.1812, disclosed - the pooled row carries the comparison, as in
> gen35): the value past K=2 remains uncollectable without function approximation.
> Worst-case committing rows (best seed's marginal vs the one-shot oracle BR): K=1 1.60x,
> K=4 1.35x the one-shot v_eq (banked K=2 1.72x, K=3 1.51x) - the standing
> regime-conditional disclosure; the dynamic policy is specialised against pattern-of-life
> behaviour.

**The act's banked claim (the thesis's Act-2 instrument).** On one instance, one baseline
family and one K-axis, the two registers separate cleanly. Against the committed adversary,
SACRED tracks the naive frontier from behind (exactly optimal stacks at K=1), narrows
monotonically to a tie as the budget saturates, and the game itself extinguishes
randomisation at K ~ 8-9; tabular FP with the same oracle stays the best single-instance
mixer past the wall. Against the observant adversary the ordering flips: the trained policy
ties the best rule at K=2 and beats every rule at K=3 AND K=4 with widening margin, to the
exact wall past which no computable adversary preserves the game; a matched-budget tabular
learner collects none of it. Both axes now end at measured laws, not at budget exhaustion.
Reported rows artefact: `models/runs/gen43_unified/reported_rows.json`
(`scratch/gen43_reported_rows.py`).

**Suite after the writer repair:** raw tail "171 passed, 3671 warnings in 16.08s"
(pre-registration run at `261703c` was "171 passed, 3671 warnings in 17.29s").
