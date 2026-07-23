#!/bin/bash
# gen34_hidden_adversary batch (pre-registered: experiments/gen34_hidden_adversary.md;
# Kilian's full launch control 2026-07-23). 3 seeds 3-parallel, then the no-intel causal
# control. Pause/resume: pkill -STOP/-CONT -f train_family_generalist.py
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
OUT=models/runs/gen34_hidden_adversary
mkdir -p "$OUT"
echo "[gen34] start $(date) sha=$(git rev-parse --short HEAD)"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_family_generalist.py --sorties 12000 --eval-every 500 \
    --seed $S --threads 3 --json-out $OUT/seed$S.json --ckpt-dir $OUT/seed${S}_ckpts \
    > $OUT/seed$S.log 2>&1 &
done
wait
echo "[gen34] seeds done $(date)"
PYTHONPATH=. $PY scripts/train_family_generalist.py --sorties 12000 --eval-every 500 \
  --no-intel --seed 0 --threads 6 --json-out $OUT/nointel_seed0.json \
  --ckpt-dir $OUT/nointel_seed0_ckpts > $OUT/nointel_seed0.log 2>&1
echo "[gen34] control done $(date)"
echo "[gen34] ALL DONE $(date)"
