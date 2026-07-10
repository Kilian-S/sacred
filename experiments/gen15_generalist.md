# Generation: gen15_generalist (A1: the map-conditioned generalist policy; ZST step 1)

- **status: PRE-REGISTERED 2026-07-10 (expansion-programme launch authority); auto-launches after
  gen14/C3 completes (chained orchestrator); binding now.**
- **git SHA:** the commit landing this ledger + the A1 machinery.

## Question (fixed before looking)

Can ONE adversarially-trained policy, conditioned on the instance (observable threat map +
per-route transferable features), learn mixed-strategy fleet routing that transfers ZERO-SHOT to
held-out OD pairs, scored against each pair's own oracle equilibrium? (The aim-level ZST promise;
the mechanism the measured ZST-0 negative said was missing.)

## Design (locked; the addendum's constraints + one recorded self-correction)

- **Conditioning:** edge-vulnerability observation column (featurise edge col 4; agents built at
  `edge_in_dim=5`) + per-route `[cost, worst-vulnerability]` features (per-instance min-max, so
  scale-free) with ONE shared learned weight vector at policy + critic heads
  (`--head-term-lr 3e-2`, the gen11b recipe). **NO `route_bias`** (identity does not transfer).
- **Per-transition menus:** route menus + features ride ON each stored transition
  (`menu_route_node_idx`/`menu_route_feats` state keys; `sac.py` per-sample plumbing), so replayed
  instance-i transitions are always scored under instance i's menu. Single-instance paths are
  unchanged (suite 155 green; the keys carry the same objects the net attributes held).
- **Adversary (SELF-CORRECTION on the addendum, recorded there too):** per-instance SMOOTH
  fictitious play (each instance keeps its own trailing occupancy window, softmax BR tau 0.05
  recomputed and sampled fresh per sortie). NOT a fixed equilibrium mixture, which is
  exploitable-by-indifference (any support route is payoff-equal against it: the B2-P2 parking
  failure in disguise).
- **Instances:** pool sampled by the F3/screen recipe (deg>=3 ODs, 3-6 disjoint base routes, k8
  menus, R in [10,14], eq >= 0.05), band 0.15-0.95, N=3, K=1; **pool-seed FIXED at 0** so every
  seed sees the same **16-train / 6-test split by OD**; test ODs are ZERO-SHOT (never trained on,
  never used for selection of anything but the best checkpoint by the primary itself).
- **Config:** fleet-route, role alphas (floor 0.20, ent-frac 0.5/0.05, follower-warmup 250),
  stack-dup 4, 12,000 sorties, eval-every 500, exact per-instance evaluation, per-eval ckpts,
  seeds {0,1,2}, `--threads 3` at 3-parallel.

## Decision metric (PRE-REGISTERED)

Per eval: for each instance, the exact policy occupancy distribution; per-instance TAP = mean of
the last TAP_K=3 exact distributions; per-instance ratio = TAP exploitability / that instance's
equilibrium. **PRIMARY = the held-out (6-OD) MEAN best-checkpoint TAP ratio, mean +/- std over 3
seeds** (best checkpoint = the eval with the lowest held-out mean, the standing minimax
discipline; drift disclosed).

> **PASS:** held-out mean best-checkpoint ratio <= 2.0 AND every held-out OD's absolute TAP
> exploitability at that checkpoint < its own loss_det (i.e. the zero-shot policy beats the
> deterministic-class optimum on EVERY unseen instance). **STRONG:** held-out mean ratio <= 1.5.
> Context anchors: single-instance post-fix training reaches ~1.3x on favourable instances
> (gen13); the un-conditioned specialist transferred at ~2.1x its holdout's equilibrium AND lost
> to random-init (ZST-0). Any PASS is the first trained evidence for the aim's ZST promise; a
> near-miss with the train-set ratio low is reported as generalisation gap (also informative).

Secondaries: train-set mean ratio (the gap read); per-OD held-out ratios; `route_feat_w`
trajectory (transferable-mechanism telemetry); alpha trajectories.

## Commands (pinned; via `scratch/gen15_chain.sh`, waits for gen14 DONE, then 3 seeds 3-parallel)

```bash
PYTHONPATH=. .venv/bin/python scripts/train_generalist.py \
  --n-train 16 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
  --head-term-lr 3e-2 --seed $S --threads 3 \
  --json-out models/runs/gen15_generalist/seed$S.json \
  --ckpt-dir models/runs/gen15_generalist/seed${S}_ckpts
```

## RESULT (to be appended)
