#!/bin/bash
# gen39 step 3: the 12-run batch (3 curricula x 3 seeds + blinded llm arm x 3 seeds), LOCAL M4,
# detached, full capacity (12 processes, threads=1 each, all maths pools capped: the standing
# multi-process dogma). Launch record in experiments/gen39_concealment.md; Kilian's launch
# authority for steps 1-3 granted in-conversation 2026-07-26.
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step3
mkdir -p "$OUT"
PY=../sacred/.venv/bin/python
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1"
for ARM in llm random heuristic; do
  for S in 0 1 2; do
    eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm $ARM --seed $S --threads 1 \
      --json-out $OUT/${ARM}_seed$S.json --ckpt-dir $OUT/${ARM}_seed${S}_ckpts \
      > $OUT/${ARM}_seed$S.log 2>&1 &"
  done
done
for S in 0 1 2; do
  eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm llm --blind --seed $S --threads 1 \
    --json-out $OUT/llmblind_seed$S.json --ckpt-dir $OUT/llmblind_seed${S}_ckpts \
    > $OUT/llmblind_seed$S.log 2>&1 &"
done
echo "launched 12 runs, logs under $OUT/"
