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

**Standing numbers after gen10 (recommendation, Kilian to confirm):** single-convoy headline =
gen10-SC **0.276** (supersedes 0.362); multi-convoy headline = the banked pre-fix best-checkpoint
(exact re-eval **0.295 +/- 0.024** at SHA `ad70a9c`) with the representation caveat disclosed,
UNTIL a post-fix multi-convoy run matches or beats it. **Proposed diagnostic (ONE run, needs
Kilian's go, pre-register before launch): gen10-MC2 = the MC config at 2400 sorties (seed 2 was
still descending at cutoff) x 3 seeds, with the role-alpha target fix behind a flag set OFF, so
the remaining confound (role-alpha targets vs menu-head discriminability) is isolated; if it
recovers <= 0.295 the post-fix number supersedes; if not, the menu head needs a discriminability
fix (e.g. a per-route learned embedding or route-cost/vulnerability features at the head) and that
is a design decision, not a knob.**
