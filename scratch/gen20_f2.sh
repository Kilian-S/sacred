#!/bin/bash
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen20_f2; mkdir -p "$OUT"; PY=.venv/bin/python
echo "[gen20] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_f2.py --od 35-159 --N 3 --K 1 --k-extra 8 --band 0.15,0.95 \
    --sorties 3000 --eval-every 200 --seed $S --threads 3 \
    --json-out "$OUT/seed$S.json" --ckpt-dir "$OUT/seed${S}_ckpts" > "$OUT/seed$S.log" 2>&1 &
done
wait
echo "[gen20] ALL DONE  $(date)" >> "$OUT/orchestrator.log"; touch "$OUT/DONE"
