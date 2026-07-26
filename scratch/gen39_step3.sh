#!/bin/bash
# gen39 step 3: the 12-run batch in TWO WAVES OF SIX (restaged 2026-07-26). The 12-way launch
# thrashed: ~1.2-3.3 GB per process on a 24 GB machine drove 16 GB of swap and the paging showed
# up as ~60-67% SYSTEM time (measured; the wait-policy env alone did not cure it). Six runs fit
# in RAM with headroom and 10 cores stay under-committed, so each wave runs at full speed.
# Same seeds, same shared oracle cache, same pinned bars: the restage changes nothing scientific.
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step3
mkdir -p "$OUT"
PY=../sacred/.venv/bin/python
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0"

run_one () {  # ARM SEED EXTRA TAG
  eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm $1 --seed $2 $3 --threads 1 \
    --json-out $OUT/$4_seed$2.json --ckpt-dir $OUT/$4_seed$2_ckpts \
    > $OUT/$4_seed$2.log 2>&1 &"
}

echo "[wave 1] llm/random/heuristic seeds 0,1"
for ARM in llm random heuristic; do
  for S in 0 1; do run_one $ARM $S "" $ARM; sleep 20; done
done
wait
echo "[wave 1] done, launching wave 2"
for ARM in llm random heuristic; do
  run_one $ARM 2 "" $ARM; sleep 20
done
for S in 0 1 2; do run_one llm $S "--blind" llmblind; sleep 20; done
wait
echo "[all 12 runs complete]"
