# Generation: gen35_dyn_kboundary (Phase-1 frontier hardening, point 2: the dynamic register where the budget bites - the first pre-registerable "beats every two-line rule" cell)

**status: PRE-REGISTERED 2026-07-23.** Mandate: Kilian's 2026-07-23 Phase-1 direction (point 2:
combine the two axes where learning already pays - within-episode dynamism and attacker
coverage). Ledger committed before any new trainer code (none is needed: the gen19 trainer
already exposes `--K`); results appended below the RESULTS line; nothing above it changes.
Training launch requires Kilian's explicit go.

**git SHA:** the commit landing this ledger; steps pin their own SHAs.

## Why (and the corrected yardstick this act stands on)

The static K-boundary act (gen26) found learning's value grows with coverage fraction but never
beats the best naive stack outright; the dynamic acts (gen19/27) beat every STATIC object but
concede that composed two-line dynamic rules sit at or ahead of the trained policy. The design
probe for this act (`scratch/gen35_kdyn_probe.py` + `scratch/gen35_mmc_check.py`, oracle-exact,
2026-07-23) mapped the corner where BOTH escape routes close, and en route exposed and repaired
a defect in the programme's dynamic yardstick:

- **Yardstick repair (binding for every dynamic number cited anywhere from now on):** the
  undamped RVI behind `oracle_refs`' `history_opt` oscillates on this deterministic-transition
  MDP class and is wrong on every cell tested (both directions, up to -57%). The exact optimum
  is the Karp minimum mean cycle, cross-checked by damped RVI (`scratch/dyn_exact.py`; the two
  agree to 5 decimals on 10/10 cells). The aerial branch found and fixed the same defect on
  2026-07-17 (`dbf385d`) - the fix never reached this branch. Corrected-yardstick appendices:
  gen19/gen27 ledgers, 2026-07-23 (their PRIMARY claims unaffected; optimum-ratio rows
  restated).
- **The corrected landscape** (`models/runs/gen35_kdyn_probe.json`,
  `models/runs/gen35_mmc_check.json`; N=3, band (0.15,0.95), k8, w=3, tau=0.15):

| instance | K | exact optimum (mmc) | best naive dynamic rule | naive/optimum | iid_eq | iid_eq/optimum |
|---|---|---|---|---|---|---|
| 35-159 (m=4) | 1 | 0.0413 | rotation 0.0413 | **1.00 (rotation IS optimal)** | 0.1468 | 3.6 |
| 35-159 (m=4) | 2 | 0.0965 | rotation 0.0965 | **1.00** | 0.2673 | 2.8 |
| 35-159 (m=4) | 3 | 0.1674 | rotation 0.1674 | **1.00** | 0.3723 | 2.2 |
| 71-33 (m=6, R=11) | 1 | 0.0313 | rotation 0.0387 | 1.31 | 0.0967 | 3.1 |
| 71-33 (m=6) | 2 | 0.0657 | rotation 0.0929 | **1.44** | 0.1823 | 2.8 |
| 71-33 (m=6) | 3 | 0.1018 | rotation 0.1539 | **1.52** | 0.2549 | 2.5 |
| 71-33 (m=6) | 4 | 0.1386 | rotation 0.2152 | **1.56** | 0.3117 | 2.2 |

  (62-97 mirrors 35-159: rotation optimal-to-1% at every K. Anti-repeat-disjoint ~= rotation
  everywhere; anti-repeat-full-menu far worse; static disjoint stacks and iid_eq worse still -
  full rows in the JSON.)

**The scoping fact (a finding in itself, already folded into the gen19 appendix):** on m=4
instances plain rotation attains the exact optimum at EVERY K - there is no dynamic-learning
headroom there, full stop. On the m=6 instance the naive-rule gap OPENS and WIDENS with K
(1.31x -> 1.56x) while dynamism keeps paying (iid_eq 2.2-3.1x the optimum). K in {2,3} on
71-33 is therefore the first regime in the whole programme where a pre-registered "the trained
policy beats every two-line rule" sentence is even POSSIBLE: the rules leave 44-52% on the
table, the optimum is exactly computable, and the optimal window cycles are non-trivial
(they use shared-edge routes the rotation family ignores).

## The game (pinned)

gen19 mechanics verbatim (`scripts/train_b1lite1.py`, UNMODIFIED - it already takes `--K`):
single instance 71-33 (kaliningrad default graph), N=3 fleet-route stacked, band (0.15,0.95),
k_extra=8, mission objective, interception_loss 10.0, w=3, tau=0.15, episode 40 sorties,
gamma 0.95. Two pre-registered cells: **K=2** and **K=3** (n_isets 903 / 12341, exact
throughout). K=4 is an OPTIONAL stretch cell (exact still feasible, 123k isets), only if both
primary cells resolve cleanly and Kilian funds it.

## Baseline family (pre-registered, non-negotiable; all exact, `scratch/dyn_exact.py` +
## `scratch/gen35_kdyn_probe.py`, values pinned above)

Naive-dynamic: best-of-20-orders disjoint rotation, anti-repeat over disjoint, anti-repeat over
full menu. Static: iid_eq mixture, uniform/inv-vuln disjoint stacks, best fixed route
(static_det). Fitted (disclosed): the exact optimal cycle itself (Karp - the "knows-the-model"
cap; any model-fitted rule is bounded by it). No-net learner row (REPORTED, not gated): tabular
average-cost Q-learning over window states with the same interaction budget as SACRED (the
gen26 lesson applied to the dynamic register; ~30 lines, built with the eval harness).
Worst-case committing row (REPORTED): best-checkpoint marginal mixture vs the one-shot oracle
BR at that K, beside the one-shot v_eq (0.2553 K=2 / 0.3829 K=3).

## Decision metric (PRE-REGISTERED)

Per cell: stationary per-sortie expected loss of the best checkpoint (TAP discipline as gen19;
analytic estimator, 2000-sortie eval), 3 seeds.

> **PRIMARY (per cell): best-checkpoint value < the best naive rule (0.0929 at K=2; 0.1539 at
> K=3) on >= 2/3 seeds AND pooled.** The first cell(s) in the programme where beating every
> naive baseline is the pre-registered bar rather than a post-hoc concession.
> **STRONG (per cell): pooled <= 1.15x the exact optimum (<= 0.0756 at K=2; <= 0.1171 at K=3).**
> **CAUSAL CONTROL: `--no-window` at K=3, expected ~iid_eq 0.2549** (the gen19 control pattern:
> the gain must be the window conditioning).
> **Branches (all writable):** PASS both cells = learning collects what naive dynamic play
> leaves on the table exactly where the static act (gen26) showed learning's value grows -
> the boundary map gains its dynamic column. PASS one = the boundary sits between K=2 and K=3;
> report as measured. FAIL = the honest dynamic analogue of gen26's tie: even 44-52% naive
> slack is not collectable by self-play at this scale; the corrected-yardstick landscape
> (pinned above) stands as the act's contribution.

## Design decisions ledgered

1. Single-instance register (gen19's), not the generalist: the claim under test is "collectable
   at all", the cheapest decisive form; zero-shot transfer of the high-K dynamic policy is
   future work after (and only after) this resolves.
2. 71-33 chosen by the probe's naive-gap screen (heuristic-gap criterion per the R0c lesson),
   NOT by det/eq; 35-159/62-97 are pinned as the scoping negatives (rotation optimal there).
3. No trainer changes: `train_b1lite1.py --od 71-33 --K {2,3}` as-is; its internal
   `oracle_refs` JSON rows are known-defective and are NOT citable - all cited refs come from
   the committed probe JSONs (disclosed here, avoiding any code change to a banked trainer).
4. Eval sorties 2000 (vs gen19's 1000): the K=3 cost surface is rougher; estimator noise must
   stay well under the PRIMARY margin.
5. Anti-chase: 8000 sorties/seed as gen19; no budget extension without a fresh pre-registered
   amendment BEFORE results are read.
6. Numbers live only in this ledger + its JSONs.

## Commands (pinned; launch = Kilian's explicit go)

```bash
# per cell K in 2 3, seeds 0 1 2 (3-parallel), full thread caps:
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  scripts/train_b1lite1.py --od 71-33 --K $K --sorties 8000 --eval-every 500 --seed $S \
  --threads 3 --json-out models/runs/gen35_dyn_kboundary/K${K}_seed$S.json \
  --ckpt-dir models/runs/gen35_dyn_kboundary/K${K}_seed${S}_ckpts
# control: --no-window --K 3 --seed 0
```

## Compute envelope

gen19 measured ~1.9 h for 3 seeds x 8000 sorties 3-parallel; K=3 adds the larger softmax
column count (~12k isets: adversary+eval cost up, SAC update unchanged) - estimate 2-4 h per
cell's seed batch. Plan: both cells + control in one night. Hard ceiling: 2 nights including
the optional K=4 stretch.

## RESULTS (appended per step; nothing above changes after launch)

**LAUNCH RECORD (2026-07-23 16:41 BST, authority: Kilian's full launch control granted
in-conversation 2026-07-23).** Batch `scratch/gen35_batch.sh` at SHA `5af4dd1` (ledger SHA;
no code changed since pre-registration): K=2 seeds 0-2 (3-parallel) -> K=3 seeds 0-2 ->
no-window control at K=3. Plumbing smokes at both K (80 sorties, seed 99, discarded for
selection purposes; `smoke_K*.json`) reproduced the probe anchors exactly (K=2 iid_eq 0.182,
static_det 0.490; K=3 0.260/0.552). Estimated completion ~2-2.5 h.

### RESULT (2026-07-23 21:41 BST; batch `scratch/gen35_batch.sh`, launch SHA `5af4dd1`;
### artefacts `models/runs/gen35_dyn_kboundary/` incl. per-eval checkpoints; scored against the
### PINNED probe refs; the trainer's internal iid_eq prints (0.257-0.258) differ from the pinned
### 0.2549 by the disclosed LP-degeneracy wobble and are not used)

| cell | seed bests | pooled | bar (best naive rule) | PRIMARY | vs exact opt | STRONG (<=1.15x) |
|---|---|---|---|---|---|---|
| K=2 | 0.0933 / 0.0919 / 0.0950 | 0.0934 | 0.0929 | **FAIL: 1/3, pooled +0.5% = a TIE AT THE RULE** | 1.42x | not met |
| K=3 | 0.1356 / 0.1428 / 0.1435 | **0.1406** | 0.1539 | **PASS: 3/3 seeds AND pooled (-8.6%)** | 1.38x | not met |

**Causal control (no-window, K=3):** best 0.2328, final iterates 0.26-0.29; the window weight
rw[2] stayed pinned at 0.00 throughout. 0.2328 = 0.91x the pinned iid_eq - as expected for a
window-blind LEARNER, which optimises the best static mixture (gen27's static rows showed the
local static optimum sits slightly below the equilibrium mixture's iid value); it remains
2.29x the exact optimum, versus the sighted arms' 1.38x. **The causal clause holds: the gain
is the window conditioning.**

> **VERDICT (per the pre-written branches): PASS AT K=3, TIE AT K=2 - "the boundary sits
> between K=2 and K=3, report as measured."** K=3 is the programme's FIRST cell where "the
> trained policy beats every naive baseline" holds under a bar fixed before launch: all three
> seeds sit 7-12% below the best two-line rule, in the regime where those rules leave 52% on
> the table. At K=2 the policy CONVERGES TO the rule (rw[2] strongly negative = the anti-repeat
> form rediscovered; pooled within 0.5% of the rule value). STRONG missed at both cells: the
> policy collects ~26% of the rule-to-optimum slack at K=3 ((0.1539-0.1406)/(0.1539-0.1018)),
> i.e. it goes meaningfully beyond every naive rule but remains far from the exact optimal
> cycle. Claim licensed: *at high interdiction budget on the m=6 instance, self-play learning
> collects value no two-line rule reaches; the exact optimum remains ~1.4x away.* The
> REPORTED rows (tabular window-Q with matched budget; worst-case committing row) follow in an
> appended record.
