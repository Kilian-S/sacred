#!/bin/bash
# gen15_generalist (A1) chained launcher: waits for gen14/C3, then 3 seeds at 3-parallel.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen15_generalist
mkdir -p "$OUT"
PY=.venv/bin/python
until [ -f models/runs/gen14_evidence/DONE ]; do sleep 120; done
echo "[gen15] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_generalist.py \
    --n-train 16 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
    --head-term-lr 3e-2 --seed $S --threads 3 \
    --json-out "$OUT/seed$S.json" \
    --ckpt-dir "$OUT/seed${S}_ckpts" > "$OUT/seed$S.log" 2>&1 &
done
wait
echo "[gen15] ALL DONE  $(date)" >> "$OUT/orchestrator.log"
touch "$OUT/DONE"
