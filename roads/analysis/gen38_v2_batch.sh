#!/bin/bash
# gen38 V2 batch (gated-in: V1 PASSED). Type-conditioned SACRED, 3 seeds. Launch ONLY when the
# machine is free (gen37 done) to avoid oversubscription. Kilian's full launch control 2026-07-24.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=.venv/bin/python
OUT=models/runs/gen38_llm_enemy_id
echo "[v2] start $(date) sha=$(git rev-parse --short HEAD)"
for S in 0 1 2; do
  PYTHONPATH=. $PY analysis/gen38_v2_conditioned.py --sorties 12000 --eval-every 1000 --seed $S \
    --threads 3 --json-out $OUT/v2_seed$S.json --ckpt-dir $OUT/v2_seed${S}_ckpts \
    > $OUT/v2_seed$S.log 2>&1 &
done
wait
echo "[v2] ALL DONE $(date)"
