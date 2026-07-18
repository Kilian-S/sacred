#!/bin/zsh
# gen29 multi-OD: 3 sighted seeds + 1 blinded control; all pools capped + niced (system-load dogma).
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
mkdir -p models/runs/gen29_multiod
for S in 0 1 2; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=. nohup nice -n 10 $PY scripts/train_multiod_generalist.py \
    --sorties 14000 --eval-every 1000 --seed $S --threads 2 \
    --json-out models/runs/gen29_multiod/seed$S.json \
    --ckpt-dir models/runs/gen29_multiod/seed${S}_ckpts \
    > models/runs/gen29_multiod/seed$S.log 2>&1 &
  disown
done
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  PYTHONPATH=. nohup nice -n 10 $PY scripts/train_multiod_generalist.py \
  --sorties 14000 --eval-every 1000 --seed 0 --blind --threads 2 \
  --json-out models/runs/gen29_multiod/blind_seed0.json \
  --ckpt-dir models/runs/gen29_multiod/blind_seed0_ckpts \
  > models/runs/gen29_multiod/blind_seed0.log 2>&1 &
disown
echo "3 sighted seeds + 1 blinded control detached -> models/runs/gen29_multiod/"
