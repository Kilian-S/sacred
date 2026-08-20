#!/bin/bash
# gen43_unified_kboundary dynamic-arm extension: K=5 then K=6 on 71-33 with k-extra 8,
# 3 seeds 3-parallel each, against the exact softmax adversary. K=6 pays roughly 25 minutes
# of startup per process.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
OUT=models/runs/gen43_unified
mkdir -p "$OUT"
echo "[gen43-ext] start $(date) sha=$(git rev-parse --short HEAD)"

for K in 5 6; do
  for S in 0 1 2; do
    PYTHONPATH=. $PY scripts/train_b1lite1.py --od 71-33 --K $K --sorties 8000 \
      --eval-every 500 --seed $S --threads 3 \
      --json-out $OUT/dyn_K${K}_seed$S.json \
      --ckpt-dir $OUT/dyn_K${K}_seed${S}_ckpts \
      > $OUT/dyn_K${K}_seed$S.log 2>&1 &
  done
  wait
  echo "[gen43-ext] dynamic K=$K done $(date)"
done

echo "[gen43-ext] ALL DONE $(date)"
