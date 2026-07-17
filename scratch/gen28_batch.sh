#!/bin/zsh
# gen28 aerial generalist: 3 seeds at 3-parallel (ledger experiments/gen28_aerial.md).
# Thread caps per the 2026-07-16 dogma; detached per the gen05 lesson. Run FROM the worktree:
#   cd /Users/kilian/Kilian/ICL/Thesis/code/sacred-aerial && zsh scratch/gen28_batch.sh
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
mkdir -p models/runs/gen28_aerial
for S in 0 1 2; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. nohup $PY \
    scripts/train_aerial_generalist.py --sorties 12000 --eval-every 500 --seed $S --threads 3 \
    --json-out models/runs/gen28_aerial/seed$S.json \
    --ckpt-dir models/runs/gen28_aerial/seed${S}_ckpts \
    > models/runs/gen28_aerial/seed$S.log 2>&1 &
  disown
done
echo "3 seeds detached; logs models/runs/gen28_aerial/seed{0,1,2}.log"
