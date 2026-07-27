#!/bin/bash
# gen39 step 5: 4 arms x 2 seeds = 8 runs, in TWO WAVES OF FOUR (the measured memory-safe shape:
# ~3.1 GB/run, 4 concurrent stays clear of the compression threshold). Wave 1 carries ALL FOUR
# arms at seed 0, so a complete cross-arm comparison is readable after wave 1.
# Staged seeds pinned in the ledger: a third seed only if the ordering is AMBIGUOUS.
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step5
mkdir -p "$OUT"
PY=../sacred/.venv/bin/python
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0"
run_one () {   # ARM SEED
  eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm $1 --seed $2 --threads 1 \
    --sorties 5000 --json-out $OUT/$1_seed$2.json --ckpt-dir $OUT/$1_seed$2_ckpts \
    > $OUT/$1_seed$2.log 2>&1 &"
  sleep 20
}
echo "[wave 1] all four arms, seed 0"
for ARM in llm16 local16 random16 tuned; do run_one $ARM 0; done
wait
echo "[wave 2] all four arms, seed 1"
for ARM in llm16 local16 random16 tuned; do run_one $ARM 1; done
wait
echo "[step 5 complete: 8 runs]"
