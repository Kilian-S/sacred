#!/bin/bash
# gen25 (A4): vanilla generalist seeds 1,2 (gen21 -> n=3) + DR generalist seed 0; 3-parallel.
set -u
mkdir -p models/runs/gen25_dr
run() {
  PYTHONPATH=. .venv/bin/python scripts/train_generalist.py \
    --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
    --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
    --head-term-lr 3e-2 $1 --seed $2 --threads 3 \
    --json-out models/runs/gen25_dr/$3.json \
    --ckpt-dir models/runs/gen25_dr/$3_ckpts \
    > models/runs/gen25_dr/$3.log 2>&1
}
run --vanilla 1 vanilla_seed1 &
run --vanilla 2 vanilla_seed2 &
run --dr 0 dr_seed0 &
wait
echo DONE > models/runs/gen25_dr/DONE
