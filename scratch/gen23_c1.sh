#!/bin/bash
# C1 (gen23): ERB-from-ALNS ablation, {seeded, cold} x 3 seeds. Chained after gen22.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen23_c1; mkdir -p "$OUT"; PY=.venv/bin/python
until [ -f models/runs/gen22_rotation/DONE ]; do sleep 120; done
echo "[gen23] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_c1.py --od 35-159 --sorties 1200 --eval-every 100 --seed $S \
    --erb --threads 3 --json-out "$OUT/seeded_seed$S.json" > "$OUT/seeded_seed$S.log" 2>&1 &
done
wait
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_c1.py --od 35-159 --sorties 1200 --eval-every 100 --seed $S \
    --threads 3 --json-out "$OUT/cold_seed$S.json" > "$OUT/cold_seed$S.log" 2>&1 &
done
wait
echo "[gen23] ALL DONE  $(date)" >> "$OUT/orchestrator.log"; touch "$OUT/DONE"
