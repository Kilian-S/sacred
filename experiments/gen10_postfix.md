# Generation: gen10_postfix (the node-ordering fix: re-run both banked headlines on corrected representations)

- **status: PRE-REGISTERED 2026-07-09 (Kilian's explicit go: "apply the fix and launch"); binding at
  launch.** Results appended below when the runs complete.
- **git SHA:** pinned by the commit that lands this ledger together with the fix (the fix and the
  pre-registration are one commit, so the record exists regardless of outcome).

## Why (the fix under test)

`CRITIQUE_INTERDICTION.md` §5.1 (2026-07-09 audit): `featurize_state` orders node rows by
**sorted(node ids)** while every consumer (protagonist + antagonist `select_action`/`update`, both
trainers' `hop_probs`, `menu_route_node_idx`) built its node->index map from **dict insertion
order**. On the Kaliningrad graph these differ, so every network ever trained in this project read a
fixed permutation of the wrong nodes' embeddings (demonstrated: convoy at node 62 reads node 167's
row). The permutation was fixed, bijective and identical across arms/train/eval, so the banked
comparisons remain internally valid; but the GNN's spatial structure was scrambled, plausibly
inflating distance-to-equilibrium in both headlines, and it makes any transfer (ZST) claim
impossible. Fixed 2026-07-09: single-source-of-truth `node_index_map` (`src/agents/networks.py`)
used at every consumer site; regression tests `tests/test_node_ordering.py` (synthetic adversarial
ordering + real Kaliningrad envs + menu pooling). Suite **149 passed** post-fix.

Two companion corrections land in the same commit (both additive; historical paths byte-identical):
1. **Role-alpha Bellman-target fix** (`sac.py`): the entropy term of V(s') now uses the temperature
   of the decision taken AT s' (follower successors get the follower alpha; previously always the
   leader alpha). Only affects role-alpha (multi-convoy adversarial) mode; historical transitions
   lack the tag and fall back to the old behaviour.
2. **Exact fleet-route evaluation** (`train_multiconvoy.py`): in fleet-route + menu-select mode the
   occupancy distribution is now computed EXACTLY (one forward pass: the fleet stacks on the leader)
   instead of the 400-sample Monte-Carlo estimate. The gen09 exact re-evaluation
   (`scratch/gen09_exact_reeval.py`) quantified the MC noise + min-selection bias at ~+0.012 on the
   best-checkpoint TAP: the pre-fix apples-to-apples comparator is therefore the EXACT re-evaluated
   **0.295 +/- 0.024**, not the ledgered MC 0.283 +/- 0.021.

## Question (fixed before looking)

**Do the two banked headlines survive, and by how much do they improve, when the policy heads read
the correct node embeddings?** Both configs are byte-for-byte the banked commands; ONLY the fix
differs (plus the declared estimator change for the multi-convoy read).

## Arms / configs (identical to the banked runs)

1. **gen10-MC** = the gen09-HEADLINE config (62->97 k8 menu-select, band 0.15-0.95, N=3, K=1,
   fleet-route, smooth FP tau 0.05, switch-every 200, smooth-window 250, leader-ent-frac 0.5,
   leader-alpha-floor 0.20, 1200 sorties, eval-every 100, per-eval checkpoints), seeds {0,1,2},
   3-parallel `--threads 3`.
2. **gen10-SC** = the B2-P3 config (33->71, k-extra 8, route-mode walk, attacker smooth, 3000
   sorties, eval-every 250), seeds {0,1,2}, 3-parallel `--threads 3` (flag added this commit;
   B2-P3 ran 4 threads serial: a runtime setting, not a dynamics one).
3. **gen10-VAN** (reference row, 1 seed): multi-convoy default branch (vanilla + independent-sacred,
   1200 sorties, seed 0) for a post-fix vanilla TAP reference (the gen09 ladder's vanilla ~0.945 is
   a pre-fix number).

## Decision metrics (PRE-REGISTERED; same as the banked generations)

- **gen10-MC PRIMARY:** fleet-route **best-checkpoint TAP** (exact estimator; selection rule
  unchanged: the per-eval snapshot with the lowest trailing-averaged-policy exploitability under the
  oracle best-response interdictor; TAP_K=5), mean +/- population std over seeds {0,1,2}.
  Comparators: pre-fix exact 0.295 +/- 0.024; ALNS 0.699; vanilla (gen10-VAN); equilibrium 0.216.
  The single-checkpoint (min per-eval expl) reading reported alongside, as in gen09.
- **gen10-SC PRIMARY (verbatim B2-P3):** `Expl_TAP(sacred) < Expl_TAP(vanilla)` on 3/3 seeds AND
  pooled, AND `Expl_TAP(sacred) < 0.455` (uniform anchor). **STRONG:** within 0.05 of the
  equilibrium 0.167. Comparators: banked B2-P3 pooled sacred 0.362 / vanilla 0.477.
- **Recorded prediction (before results):** post-fix numbers land AT OR BELOW the pre-fix ones
  (correct embeddings can only add information): gen10-SC pooled sacred TAP <= 0.362; gen10-MC
  exact best-checkpoint TAP <= 0.295. The last-iterate drift toward uniform (the gen09 transient
  finding) is expected to PERSIST (it is an FP-dynamics property, not a representation one); if it
  disappears, that is a major finding to report. If any number comes out WORSE, it is reported as
  measured and the banked pre-fix results stand as the citable ones (pinned at their SHAs).

## Commands (pinned; via `scratch/gen10_orchestrator.sh`, detached, all outputs saved)

```bash
# stage 1: gen10-MC, seeds 0,1,2 at 3-parallel
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
  --seed $S --threads 3 --json-out models/runs/gen10_postfix/mc_seed$S.json \
  --ckpt-dir models/runs/gen10_postfix/mc_seed${S}_ckpts \
  > models/runs/gen10_postfix/mc_seed$S.log 2>&1
# stage 2: gen10-SC, seeds 0,1,2 at 3-parallel
PYTHONPATH=. .venv/bin/python scripts/train_interdiction.py \
  --k-extra 8 --route-mode walk --attacker-mode smooth --sorties 3000 --seed $S \
  --eval-every 250 --threads 3 --json-out models/runs/gen10_postfix/B2P3_seed$S.json \
  > models/runs/gen10_postfix/B2P3_seed$S.log 2>&1
# stage 3: gen10-VAN, seed 0
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --sorties 1200 --eval-every 100 --seed 0 --threads 4 \
  --json-out models/runs/gen10_postfix/van_seed0.json \
  > models/runs/gen10_postfix/van_seed0.log 2>&1
```

Timing estimate: stage 1 ~15-25 min (3-parallel), stage 2 ~50-70 min (3-parallel; B2-P3 was ~40
min/seed serial), stage 3 ~15-25 min; ~1.5-2 h wall total.

## RESULT (to be appended; nothing above this line changes after launch)

### gen10-MC RESULT (2026-07-09, stage 1, 3 seeds, ~15 min at 3-parallel): PREDICTION VIOLATED; post-fix is WORSE

| seed | best-ckpt TAP @ sortie | best single-ckpt @ sortie | final TAP | telemetry signature |
|---|---|---|---|---|
| 0 | 0.478 @ 200 | 0.564 @ 100 | 0.721 | early shallow best, drift up; alpha floors ~500 then RISES late (0.20 -> 1.20) |
| 1 | 0.409 @ 300 | 0.453 @ 100 | 0.702 | same shape |
| 2 | 0.454 @ 1200 | 0.646 @ 1200 | 0.518 | **ALPHA RUNAWAY: H_lead pinned 0.00 for ~900 sorties, alpha 1.2 -> 71; breaks free ~sortie 1000 and is still descending at cutoff** |

**Post-fix EXACT best-checkpoint TAP mean 0.447 +/- 0.029** vs the pre-fix exact comparator
**0.295 +/- 0.024**: the recorded prediction (post-fix <= pre-fix) is VIOLATED; reported as
measured. Per the pre-registration, **the banked pre-fix results stand as the citable multi-convoy
numbers** (at their pinned SHAs, with the representation caveat now disclosed in
CRITIQUE_INTERDICTION.md).

**Reading (mechanism candidates, NOT post-hoc tuned):** three changes are confounded in this arm:
(a) the representation fix itself: with CORRECT embeddings, overlapping routes now have genuinely
similar mean-pooled embeddings, so the menu head is LESS discriminative than under the bug (the
fixed permutation had been acting as an accidental route-identity hash: distinct random node sets
per route made memorisation easy); (b) the role-alpha target fix changes the leader's Bellman
targets (its bootstrap V(s') now uses the fast-collapsing follower temperature); (c) the config
(lr, floor, ent-frac, 1200 sorties) was tuned under the buggy representation. Seed 2's 900-sortie
H=0 park with alpha runaway is a softmax-saturation trap the old representation never showed.
Stage 2 (single-convoy, NO role alpha, NO menu head) isolates the representation fix alone; its
read arrives below. Any diagnostic re-run (role-alpha fix flagged off; longer horizon: seed 2 was
still descending) is a NEW pre-registration and needs Kilian's go.

### gen10-SC RESULT (2026-07-09, stage 2, 3 seeds, ~74 min at 3-parallel): PRIMARY PASSED on every clause; post-fix MARKEDLY BETTER

| seed | arm | **expl_TAP (primary)** | expl_policy | expl_avg | cost(TAP) |
|---|---|---|---|---|---|
| 0 | vanilla / sacred | **0.483 / 0.252** | 0.480 / 0.258 | 0.430 / 0.258 | 8.4 / 13.6 |
| 1 | vanilla / sacred | **0.485 / 0.261** | 0.529 / 0.387 | 0.436 / 0.272 | 8.4 / 13.3 |
| 2 | vanilla / sacred | **0.472 / 0.316** | 0.487 / 0.561 | 0.437 / 0.277 | 8.4 / 12.8 |

Anchors: shortest_path 1.000 @ 4.1; uniform 0.455 @ 12.4; equilibrium 0.167 @ 16.0.

- **PRIMARY PASS, every clause:** sacred < vanilla on 3/3 seeds; **pooled sacred 0.276 vs vanilla
  0.480**; all seeds < 0.455. The B2-P3 primary REPLICATES post-fix.
- **The prediction (post-fix <= pre-fix) is CONFIRMED for this arm, and the improvement is large:
  pooled sacred TAP 0.276 vs the banked 0.362 (distance-to-equilibrium 0.109 vs 0.195, i.e. ~44%
  of the residual gap was the representation bug).** Vanilla is unchanged (0.480 vs 0.477), as
  expected: cost-driven mixing does not depend on embedding quality. STRONG form (<= 0.05 of
  0.167) still not met (best seed 0.085) but materially closer.
- **Reading:** the clean isolation worked. The representation fix ALONE (walk mode: no role alpha,
  no menu head) improves the equilibrium approach substantially; the multi-convoy regression in
  stage 1 is therefore localised to the menu-head discriminability under correct embeddings and/or
  the role-alpha target change and/or the pre-fix-tuned config, NOT to the ordering fix itself.
- **Citable status:** gen10-SC (this SHA) SUPERSEDES B2-P3's 0.362 as the single-convoy headline
  number, pending Kilian's confirmation: same instance, same pre-registered metric, every clause
  passed, strictly better, produced on corrected representations. Ladder (TAP, pooled): shortest
  1.000 > vanilla 0.480 > uniform 0.455 > **sacred 0.276** >> equilibrium 0.167.

### gen10-VAN RESULT (2026-07-09, stage 3, seed 0): post-fix vanilla reference for the multi-convoy ladder

Post-fix vanilla (independent convoys, travel objective) TAP **0.859** (pre-fix ~0.945); the
sacred-independent secondary lands 0.782 (expected weak: independent routing cannot reach the
correlated optimum, the M3 finding replicating post-fix).

### gen10 OVERALL (all stages complete, 2026-07-09 23:01)

**Post-fix multi-convoy ladder as measured (exact estimator):** shortest 0.973 > vanilla 0.859 >
ALNS 0.699 > **SACRED fleet-route best-ckpt 0.447 +/- 0.029** > equilibrium 0.216. Note the
qualitative Obj-5 ordering SURVIVES the fix on every seed (post-fix SACRED still beats ALNS by
0.25 and vanilla by 0.41); what regressed is the MARGIN toward the equilibrium relative to the
banked pre-fix 0.295 exact / 0.283 MC.

**Single-convoy:** post-fix PASS on every clause and a large improvement (pooled 0.276 vs banked
0.362; ~44% of the residual equilibrium gap was the representation bug). Vanilla unchanged.

**Standing numbers after gen10 (CONFIRMED by Kilian 2026-07-10): the single-convoy headline is
gen10-SC 0.276 (SUPERSEDES B2-P3's 0.362)**; multi-convoy headline = the banked pre-fix
best-checkpoint (exact re-eval **0.295 +/- 0.024** at SHA `ad70a9c`) with the representation
caveat disclosed, UNTIL a post-fix multi-convoy run matches or beats it.

## gen10-MC2 pre-registration (2026-07-10, Kilian's explicit go; binding at launch)

**Purpose:** attribute the gen10-MC regression and attempt the post-fix recovery. TWO deliberate
changes vs gen10-MC, declared up front (a pragmatic recovery run, not a factorial isolation; the
attribution reads below are partial and stated as such):
1. **`--legacy-role-target`** (new flag, this commit): reverts the role-alpha TARGET fix (V(s')
   entropy term uses the primary alpha, the pre-fix behaviour) while KEEPING the node-ordering
   fix. Removes the target-scale confound.
2. **`--sorties 2400`** (was 1200): gen10-MC seed 2 was still descending at cutoff, and seeds 0/1
   peaked earlier/shallower than pre-fix, so the post-fix timescale may simply be longer.

Everything else IDENTICAL to gen10-MC (62->97 k8 menu-select, band 0.15-0.95, N=3, K=1,
fleet-route, smooth tau 0.05, switch-every 200, window 250, leader-ent-frac 0.5, floor 0.20,
eval-every 100, EXACT estimator, per-eval checkpoints), seeds {0,1,2}, 3-parallel `--threads 3`.

**Decision reading (pre-committed):** primary = exact best-checkpoint TAP mean +/- pop std.
- **<= 0.295:** post-fix recovery achieved; this number SUPERSEDES the pre-fix multi-convoy
  headline (pending Kilian's confirmation), and the regression is attributed to the target-fix
  and/or horizon (attribution partial, stated).
- **0.295-0.447:** partial recovery; the pre-fix 0.295 stands; the residual points at menu-head
  discriminability under correct embeddings; the next step is a DESIGN change (route-level
  features at the head), proposed separately, not more knobs.
- **>= 0.447:** no recovery; same consequence as above, stronger.
Alpha trajectories + H_lead reported per seed (the runaway/park signature is the secondary read).

**Command (pinned; per-seed via `scratch/gen10_mc2.sh`, all saved under `models/runs/gen10_postfix/`):**
```bash
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 2400 --eval-every 100 \
  --legacy-role-target --seed $S --threads 3 \
  --json-out models/runs/gen10_postfix/mc2_seed$S.json \
  --ckpt-dir models/runs/gen10_postfix/mc2_seed${S}_ckpts \
  > models/runs/gen10_postfix/mc2_seed$S.log 2>&1
```
SHA pinned by the commit landing this pre-registration + the flag. RESULT appended below.

### FLEET-COST column + vanilla best-checkpoint row (2026-07-10 night, EVAL-ONLY; closes CRITIQUE_PREFREEZE §3.4-3.5)

`scratch/fleet_cost_probe.py` (validated: reproduces the gen09 exact re-eval TAPs 0.281/0.274/0.329
exactly, with each checkpoint era evaluated under ITS OWN indexing convention). Expected fleet
travel cost per sortie (62-97 k8, N=3; route costs 26.3-52.2):

| arm | mission-failure expl | fleet cost / sortie |
|---|---|---|
| shortest-path stack | 0.973 | 78.9 |
| ALNS plan (spread) | 0.699 | 96.1 |
| SACRED pre-fix best-ckpt (gen09, exact) | 0.295 +/- 0.024 | **123.1 +/- 1.0** |
| SACRED post-fix best-ckpt (gen10-MC) | 0.447 +/- 0.029 | 115.3 +/- 2.7 |
| equilibrium mixture | 0.216 | 120.8 |

**Read:** SACRED's security is bought at a ~28% fleet-cost premium over ALNS's spread plan, and
its premium is essentially THE EQUILIBRIUM'S OWN (123.1 vs 120.8): the randomised stack pays what
optimal play pays, not an RL-inefficiency surcharge. **Vanilla selection symmetry:** gen10-VAN's
best-checkpoint TAP is 0.806 (vs final 0.855): best-checkpoint selection does not rescue the
non-adversarial control. (Post-fix vanilla is extended to 3 seeds in the gen12 sweep batch.)

### gen10-MC2 RESULT (2026-07-10, 3 seeds, ~32 min at 3-parallel): NO RECOVERY; attribution resolved

| seed | best-ckpt TAP @ sortie | final TAP | telemetry signature |
|---|---|---|---|
| 0 | 0.442 @ 700 | 0.957 | early H=0 park (~sortie 100), alpha spike to 3.9, partial escape, late drift to near-pure |
| 1 | 0.441 @ 100 | 0.776 | best at the FIRST eval; alpha floors 0.20 then climbs to 2.1 late |
| 2 | 0.458 @ 100 | 0.707 | same shape; alpha climbs to 4.6 late |

**gen10-MC2 EXACT best-checkpoint TAP mean 0.447 +/- 0.008**: numerically identical to gen10-MC's
0.447 +/- 0.029 (and tighter). Pre-committed reading: **>= 0.447 branch = NO recovery.**

**Attribution (now clean):** reverting the role-alpha target fix changed NOTHING (0.447 with the
fix, 0.447 without), and doubling the horizon did not help (best checkpoints land at sorties
100-700; the tail drifts/parks). So the regression is attributable to the REPRESENTATION change
itself: with correct embeddings, overlapping routes have near-identical mean-pooled embeddings and
the parameter-free menu head cannot separate them the way the pre-fix accidental route-identity
hash could. The striking reproducibility (0.447 across two independent 3-seed runs under different
target rules) reads as a structural plateau of the current head, not noise.

**Consequences (per the pre-registration):** (1) the pre-fix multi-convoy headline STANDS as the
citable number (exact **0.295 +/- 0.024** at SHA `ad70a9c`), with the representation caveat
disclosed; the post-fix reproduction stands at 0.447 (still beating ALNS 0.699 and vanilla 0.859
on every seed, so the Obj-5 ordering is bug-robust). (2) The next step is a DESIGN change, not a
knob: give the menu head undiluted per-route scalar features via the proven lever-2 pattern
(learned weights on normalised route COST and route VULNERABILITY aggregates, computable from the
game/threat map, at BOTH the policy and critic heads). This restores head-level discriminability
for near-duplicate routes AND is precisely the map-conditioning ZST needs (CRITIQUE §7). Proposed
as gen11, pre-registered separately; launch = Kilian's go. Multi-convoy work otherwise closed at
this state.
