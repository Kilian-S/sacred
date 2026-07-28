#!/bin/bash
# gen36 Step A batch (pre-registered: experiments/gen36_multiod_rescue.md; Kilian's full
# launch control 2026-07-23). 3 distillation seeds, parallel.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
OUT=models/runs/gen36_multiod_rescue
echo "[gen36A] start $(date) sha=$(git rev-parse --short HEAD)"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/distill_multiod.py --seed $S --threads 2 \
    --json-out $OUT/distill_seed$S.json > $OUT/distill_seed$S.log 2>&1 &
done
wait
echo "[gen36A] ALL DONE $(date)"
