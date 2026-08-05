#!/bin/bash
# gen41 deep-window ZST batch: 3 seeds + no-window control, 4-concurrent, thread pools
# capped (SYSTEM dogma), threads 2 each on the M4 (4P+6E). Launch detached:
#   nohup bash scratch/gen41_batch.sh > models/runs/gen41_deepwindow/batch.log 2>&1 & disown
set -u
mkdir -p models/runs/gen41_deepwindow
COMMON="--pool-file models/runs/gen41_pool.json --K 2 --k-extra 12 --window 6 --fast-refs \
  --sorties 12000 --eval-every 500 --eval-n 600 --eval-n-train 250 --threads 2"
for S in 0 1 2; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
    scripts/train_dyn_generalist.py $COMMON --seed $S \
    --json-out models/runs/gen41_deepwindow/seed$S.json \
    --ckpt-dir models/runs/gen41_deepwindow/seed${S}_ckpts \
    > models/runs/gen41_deepwindow/seed$S.log 2>&1 &
done
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python \
  scripts/train_dyn_generalist.py $COMMON --no-window --seed 0 \
  --json-out models/runs/gen41_deepwindow/seed0_nowin.json \
  --ckpt-dir models/runs/gen41_deepwindow/seed0_nowin_ckpts \
  > models/runs/gen41_deepwindow/seed0_nowin.log 2>&1 &
wait
echo "gen41 batch complete"
