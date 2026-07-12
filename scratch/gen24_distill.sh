#!/bin/bash
# gen24 (A1): LP-distillation control, 3 seeds, serial (each run is minutes; sampling dominates).
set -u
mkdir -p models/runs/gen24_distill
for S in 0 1 2; do
  PYTHONPATH=. .venv/bin/python scripts/train_distill.py \
    --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
    --n-per-city 6 --n-test 6 --pool-seed 0 --steps 1500 --eval-every 100 \
    --head-term-lr 3e-2 --seed $S --threads 4 \
    --json-out models/runs/gen24_distill/seed$S.json \
    --ckpt-dir models/runs/gen24_distill/seed${S}_ckpts \
    > models/runs/gen24_distill/seed$S.log 2>&1
done
echo DONE > models/runs/gen24_distill/DONE
