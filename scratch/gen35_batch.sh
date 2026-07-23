#!/bin/bash
# gen35_dyn_kboundary batch (pre-registered: experiments/gen35_dyn_kboundary.md; launch
# authorised by Kilian 2026-07-23, full launch control). K=2 then K=3 (3 seeds, 3-parallel),
# then the no-window causal control at K=3.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
OUT=models/runs/gen35_dyn_kboundary
mkdir -p "$OUT"
echo "[batch] start $(date) sha=$(git rev-parse --short HEAD)"
for K in 2 3; do
  for S in 0 1 2; do
    PYTHONPATH=. $PY scripts/train_b1lite1.py --od 71-33 --K $K --sorties 8000 \
      --eval-every 500 --seed $S --threads 3 \
      --json-out $OUT/K${K}_seed$S.json --ckpt-dir $OUT/K${K}_seed${S}_ckpts \
      > $OUT/K${K}_seed$S.log 2>&1 &
  done
  wait
  echo "[batch] K=$K seeds done $(date)"
done
PYTHONPATH=. $PY scripts/train_b1lite1.py --od 71-33 --K 3 --no-window --sorties 8000 \
  --eval-every 500 --seed 0 --threads 3 \
  --json-out $OUT/K3_nowin_seed0.json --ckpt-dir $OUT/K3_nowin_seed0_ckpts \
  > $OUT/K3_nowin_seed0.log 2>&1
echo "[batch] control done $(date)"
echo "[batch] ALL DONE $(date)"
