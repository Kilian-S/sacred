#!/bin/zsh
# gen28 v2.3 walker batch: 3 seeds, all pools capped, niced (Kilian's system-load constraint).
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
mkdir -p models/runs/gen28_walker
for S in 0 1 2; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=. nohup nice -n 10 $PY \
    scripts/train_aerial_walker.py --sorties 12000 --eval-every 500 --seed $S --threads 2 \
    --json-out models/runs/gen28_walker/seed$S.json \
    --ckpt-dir models/runs/gen28_walker/seed${S}_ckpts \
    > models/runs/gen28_walker/seed$S.log 2>&1 &
  disown
done
echo "3 walker seeds detached"
