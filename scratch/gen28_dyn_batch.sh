#!/bin/zsh
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
mkdir -p models/runs/gen28_dyn
for S in 0 1 2; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=. nohup nice -n 10 $PY \
    scripts/train_aerial_dyn.py --sorties 16000 --eval-every 1000 --seed $S --threads 2 \
    --json-out models/runs/gen28_dyn/seed$S.json \
    --ckpt-dir models/runs/gen28_dyn/seed${S}_ckpts \
    > models/runs/gen28_dyn/seed$S.log 2>&1 &
  disown
done
echo "3 dyn seeds detached"
