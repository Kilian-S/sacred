#!/bin/bash
# gen39 step 5, THIRD SEED. Triggered by the ambiguity rule pinned in the ledger BEFORE the batch
# (any pair within 10% pooled: llm16~local16 8.2%, random16~tuned 2.8%), not by which arm lost.
# Same 4 arms, same curricula, same test set; seed 2 only. One wave of four.
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step5
PY=../sacred/.venv/bin/python
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0"
for ARM in llm16 local16 random16 tuned; do
  eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm $ARM --seed 2 --threads 1 \
    --sorties 5000 --json-out $OUT/${ARM}_seed2.json --ckpt-dir $OUT/${ARM}_seed2_ckpts \
    > $OUT/${ARM}_seed2.log 2>&1 &"
  sleep 20
done
wait
echo "[seed 2 complete]"
