#!/bin/bash
# gen43_unified_kboundary batch. Static cells K in {1,2,3,4,7,8}, 3 seeds 3-parallel each,
# with an exact attacker at K<=3 and greedy best-response at K>=4, then dynamic cells
# K in {1,4}, 3 seeds 3-parallel.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
OUT=models/runs/gen43_unified
mkdir -p "$OUT"
echo "[gen43] start $(date) sha=$(git rev-parse --short HEAD)"

for K in 1 2 3 4 7 8; do
  GB=""
  if [ "$K" -ge 4 ]; then GB="--greedy-br"; fi
  for S in 0 1 2; do
    PYTHONPATH=. $PY scripts/train_multiconvoy.py \
      --od 71-33 --N 3 --K $K --k-extra 8 --menu-select --band 0.15,0.95 \
      --fleet-route --attacker-mode smooth $GB --fp-tau 0.05 --switch-every 200 \
      --smooth-window 250 --leader-ent-frac 0.5 --leader-alpha-floor 0.20 \
      --sorties 1200 --eval-every 100 --skip-vanilla --seed $S --threads 3 \
      --json-out $OUT/static_K${K}_seed$S.json \
      --ckpt-dir $OUT/static_K${K}_seed${S}_ckpts \
      > $OUT/static_K${K}_seed$S.log 2>&1 &
  done
  wait
  echo "[gen43] static K=$K done $(date)"
done

for K in 1 4; do
  for S in 0 1 2; do
    PYTHONPATH=. $PY scripts/train_b1lite1.py --od 71-33 --K $K --sorties 8000 \
      --eval-every 500 --seed $S --threads 3 \
      --json-out $OUT/dyn_K${K}_seed$S.json \
      --ckpt-dir $OUT/dyn_K${K}_seed${S}_ckpts \
      > $OUT/dyn_K${K}_seed$S.log 2>&1 &
  done
  wait
  echo "[gen43] dynamic K=$K done $(date)"
done

echo "[gen43] ALL DONE $(date)"
