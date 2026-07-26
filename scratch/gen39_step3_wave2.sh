#!/bin/bash
# gen39 step 3, wave 2 watcher: waits until all six wave-1 result JSONs exist, then launches
# wave 2 (seed-2 arms + the three blinded runs) with the recorded nice recipe. Replaces the
# original in-script wait (that launcher was killed when the batch was reniced mid-flight).
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step3
PY=../sacred/.venv/bin/python
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0"
until [ -s "$OUT/llm_seed0.json" ] && [ -s "$OUT/llm_seed1.json" ] \
   && [ -s "$OUT/random_seed0.json" ] && [ -s "$OUT/random_seed1.json" ] \
   && [ -s "$OUT/heuristic_seed0.json" ] && [ -s "$OUT/heuristic_seed1.json" ]; do
  sleep 120
done
echo "[wave 2] wave 1 complete, launching"
run_one () {
  eval "$ENV nohup nice -n 10 $PY scripts/train_gen39_conceal.py --arm $1 --seed $2 $3 --threads 1 \
    --json-out $OUT/$4_seed$2.json --ckpt-dir $OUT/$4_seed$2_ckpts \
    > $OUT/$4_seed$2.log 2>&1 &"
}
for ARM in llm random heuristic; do run_one $ARM 2 "" $ARM; sleep 20; done
for S in 0 1 2; do run_one llm $S "--blind" llmblind; sleep 20; done
echo "[wave 2] launched"
