#!/bin/bash
# gen39 step 3: 12 runs in THREE WAVES OF FOUR (final staging, 2026-07-26).
# Measured on the repaired trainer: 1.81 s/sortie SOLO (362 s / 200 sorties), footprint ~3.1 GB
# RSS/run and stable. Four concurrent keeps the machine clear of the memory-compression
# threshold that caused every earlier crawl; 5000 sorties/run per the pre-registered amendment.
# Wave 1 deliberately carries ALL THREE ARMS at seed 0 (plus llm seed 1), so a first cross-arm
# comparison is readable after wave 1 rather than at the very end.
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step3
mkdir -p "$OUT"
PY=../sacred/.venv/bin/python
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0"
run_one () {  # ARM SEED EXTRA TAG
  eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm $1 --seed $2 $3 --threads 1 \
    --sorties 5000 --json-out $OUT/$4_seed$2.json --ckpt-dir $OUT/$4_seed$2_ckpts \
    > $OUT/$4_seed$2.log 2>&1 &"
  sleep 20
}
echo "[wave 1] llm/random/heuristic seed 0 + llm seed 1"
run_one llm 0 "" llm; run_one random 0 "" random; run_one heuristic 0 "" heuristic; run_one llm 1 "" llm
wait
echo "[wave 2] random/heuristic seed 1 + llm/random seed 2"
run_one random 1 "" random; run_one heuristic 1 "" heuristic; run_one llm 2 "" llm; run_one random 2 "" random
wait
echo "[wave 3] heuristic seed 2 + the three blinded runs"
run_one heuristic 2 "" heuristic; run_one llm 0 "--blind" llmblind; run_one llm 1 "--blind" llmblind; run_one llm 2 "--blind" llmblind
wait
echo "[all 12 runs complete]"
