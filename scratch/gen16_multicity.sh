#!/bin/bash
# gen16_multicity: train on kaliningrad+east_london+istanbul, hold out gdansk; 3 seeds 3-parallel.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen16_multicity
mkdir -p "$OUT"
PY=.venv/bin/python
echo "[gen16] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_generalist.py \
    --cities kaliningrad,east_london,istanbul --holdout-city gdansk \
    --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
    --head-term-lr 3e-2 --seed $S --threads 3 \
    --json-out "$OUT/seed$S.json" \
    --ckpt-dir "$OUT/seed${S}_ckpts" > "$OUT/seed$S.log" 2>&1 &
done
wait
echo "[gen16] ALL DONE  $(date)" >> "$OUT/orchestrator.log"
touch "$OUT/DONE"
