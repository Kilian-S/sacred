#!/bin/zsh
# gen28 aerial generalist: 3 seeds at 3-parallel (ledger experiments/gen28_aerial.md).
# Thread caps per the 2026-07-16 dogma; detached per the gen05 lesson. Run FROM the worktree:
#   cd /Users/kilian/Kilian/ICL/Thesis/code/sacred-aerial && zsh scratch/gen28_batch.sh
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
mkdir -p models/runs/gen28_fleet2
# All pools capped + torch 2 threads/seed (6 compute threads on 10 cores) + nice: Kilian's
# 2026-07-17 constraint (past runs spiked ~40% system time from uncapped pools).
for S in 0 1 2; do
  OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=. nohup nice -n 10 $PY \
    scripts/train_aerial_generalist.py --sorties 12000 --eval-every 500 --seed $S --threads 2 \
    --json-out models/runs/gen28_fleet2/seed$S.json \
    --ckpt-dir models/runs/gen28_fleet2/seed${S}_ckpts \
    > models/runs/gen28_fleet2/seed$S.log 2>&1 &
  disown
done
echo "3 seeds detached; logs models/runs/gen28_fleet2/seed{0,1,2}.log"
