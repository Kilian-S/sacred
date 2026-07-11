#!/bin/bash
# Item 2.3: leave-one-city-out rotation cell - train Kaliningrad+East London+Gdansk, HOLD OUT
# ISTANBUL (the structurally most distant city), 3 seeds, gen16 bars. Chained after F2+vanilla.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen22_rotation; mkdir -p "$OUT"; PY=.venv/bin/python
until [ -f models/runs/gen20_f2/DONE ] && { [ -f models/runs/gen21_vanilla/DONE ] || grep -q "GENERALIST (seed" models/runs/gen21_vanilla/seed0.log 2>/dev/null; }; do sleep 120; done
echo "[gen22] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_generalist.py \
    --cities kaliningrad,east_london,gdansk --holdout-city istanbul \
    --n-per-city 6 --n-test 6 --pool-seed 0 --sorties 12000 --eval-every 500 \
    --head-term-lr 3e-2 --seed $S --threads 3 \
    --json-out "$OUT/seed$S.json" --ckpt-dir "$OUT/seed${S}_ckpts" > "$OUT/seed$S.log" 2>&1 &
done
wait
echo "[gen22] ALL DONE  $(date)" >> "$OUT/orchestrator.log"; touch "$OUT/DONE"
