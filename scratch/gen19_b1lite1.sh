#!/bin/bash
# gen19_b1lite1: 3 seeds history-aware + 1 no-window causal control (seed 0). 35-159 w=3 tau=0.15.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen19_b1lite1
mkdir -p "$OUT"
PY=.venv/bin/python
echo "[gen19] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_b1lite1.py \
    --od 35-159 --N 3 --K 1 --k-extra 8 --band 0.15,0.95 --window 3 --tau 0.15 \
    --episode-len 40 --gamma 0.95 --sorties 8000 --eval-every 500 --head-term-lr 3e-2 \
    --seed $S --threads 3 --json-out "$OUT/seed$S.json" \
    --ckpt-dir "$OUT/seed${S}_ckpts" > "$OUT/seed$S.log" 2>&1 &
done
wait
echo "[gen19] history-aware done; no-window control  $(date)" >> "$OUT/orchestrator.log"
PYTHONPATH=. $PY scripts/train_b1lite1.py \
  --od 35-159 --N 3 --K 1 --k-extra 8 --band 0.15,0.95 --window 3 --tau 0.15 \
  --episode-len 40 --gamma 0.95 --sorties 8000 --eval-every 500 --head-term-lr 3e-2 \
  --no-window --seed 0 --threads 4 --json-out "$OUT/nowindow_seed0.json" \
  > "$OUT/nowindow_seed0.log" 2>&1
echo "[gen19] ALL DONE  $(date)" >> "$OUT/orchestrator.log"
touch "$OUT/DONE"
