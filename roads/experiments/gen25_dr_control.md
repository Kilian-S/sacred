# gen25: domain-randomisation control and the vanilla arm at n=3

Registered 2026-07-12. Results 2026-07-13.

Artefacts: `models/runs/gen25_dr/*.json` (`vanilla_seed{1,2}.json`, `dr_seed0.json`) and the matching `*_ckpts` directories. Script: `scripts/train_generalist.py` with the `--dr` and `--vanilla` flags.

## Question

Is best-response pressure, rather than threat exposure of any kind, the ingredient the adversarial generalist's zero-shot transfer depends on?

## Game

All arms use the gen16 recipe verbatim: cities kaliningrad + east_london + istanbul, holdout gdansk, pool-seed 0, 12000 sorties, eval-every 500, head-term lr 3e-2, per-eval checkpoints.

| arm | flag | seeds |
|---|---|---|
| vanilla generalist (gen21 extension) | `--vanilla` | {1, 2}, seed 0 being the gen21 run |
| domain-randomisation generalist | `--dr` | {0} |

The DR arm keeps the mission objective but samples the interdictor uniformly at random each sortie instead of smooth-FP best-responding.

```bash
# vanilla seeds 1,2 (gen21 config verbatim)
PYTHONPATH=. .venv/bin/python scripts/train_generalist.py \
  --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
  --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
  --head-term-lr 3e-2 --vanilla --seed $S --threads 3 \
  --json-out models/runs/gen25_dr/vanilla_seed$S.json \
  --ckpt-dir models/runs/gen25_dr/vanilla_seed${S}_ckpts
# DR seed 0
PYTHONPATH=. .venv/bin/python scripts/train_generalist.py \
  --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
  --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
  --head-term-lr 3e-2 --dr --seed 0 --threads 3 \
  --json-out models/runs/gen25_dr/dr_seed0.json \
  --ckpt-dir models/runs/gen25_dr/dr_seed0_ckpts
```

## Criteria

Metric: held-out Gdansk mean best-checkpoint TAP ratio, select-on-train, per arm. Checkpoint granularity 500.

- Vanilla at n=3: report mean and standard deviation, whatever the values.
- DR above 1.88 (outside one pooled standard deviation from gen16): best-response pressure, not threat exposure, is the causal ingredient.
- DR within [1.58, 1.88]: threat exposure suffices.
- DR below 1.58: DR transfers better than adversarial training.

## Baselines

- Adversarial generalist (gen16): 1.733 +/- 0.149 select-on-train.
- random-init: an untrained network, ~1.99.
- Distillation with validation stop, 1.555, and retrieval, 1.676; both consume train-side equilibrium labels.
- uniform-disjoint stack: 1.989 on these ODs.
- Vanilla generalist (gen21 seed 0): 2.338.

## Results

| arm | select-on-train | select-on-test | final iterate |
|---|---|---|---|
| vanilla seed 1 | 2.351 @ 500 | 2.351 | 2.573 |
| vanilla seed 2 | 2.372 @ 500 | 2.372 | 2.549 |
| vanilla, n=3 (with gen21 seed 0 at 2.338) | 2.354 +/- 0.014 | - | - |
| DR seed 0 | 2.056 @ 500 | 1.776 | 2.298 |

Vanilla at n=3 is 2.354 +/- 0.014, above the random-init reference (~1.99). The DR arm lands at 2.056 select-on-train, above the 1.88 band edge, so the pre-committed DR-above-1.88 branch is the one that fired; its select-on-test bound is 1.776, and the arm is n=1.

Transfer ladder on the held-out Gdansk ODs, select-on-train where applicable: distill+val 1.555 < retrieval 1.676 < adversarial 1.733 < DR 2.056 ~ uniform-stack 1.989 ~ random-init 1.99 < vanilla 2.354.
