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
