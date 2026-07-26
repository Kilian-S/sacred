#!/bin/bash
# gen39 step 3: the 12-run batch (3 curricula x 3 seeds + blinded llm arm x 3 seeds), LOCAL M4,
# detached, full capacity (12 processes, threads=1 each, all maths pools capped: the standing
# multi-process dogma). Launch record in experiments/gen39_concealment.md; Kilian's launch
# authority for steps 1-3 granted in-conversation 2026-07-26.
cd "$(dirname "$0")/.." || exit 1
OUT=models/runs/gen39_step3
mkdir -p "$OUT"
PY=../sacred/.venv/bin/python
# OMP_WAIT_POLICY=PASSIVE + KMP_BLOCKTIME=0: idle pool threads SLEEP instead of spin-waiting.
# The first launch omitted them and 12 procs x 7 threads of active spin showed up as ~67% SYSTEM
# time (the 2026-07-16 dogma's failure mode at larger scale). Launches staggered 45 s so the
# shapely-heavy pool builds do not storm the allocator together.
ENV="PYTHONPATH=. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0"
for ARM in llm random heuristic; do
  for S in 0 1 2; do
    eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm $ARM --seed $S --threads 1 \
      --json-out $OUT/${ARM}_seed$S.json --ckpt-dir $OUT/${ARM}_seed${S}_ckpts \
      > $OUT/${ARM}_seed$S.log 2>&1 &"
    sleep 45
  done
done
for S in 0 1 2; do
  eval "$ENV nohup $PY scripts/train_gen39_conceal.py --arm llm --blind --seed $S --threads 1 \
    --json-out $OUT/llmblind_seed$S.json --ckpt-dir $OUT/llmblind_seed${S}_ckpts \
    > $OUT/llmblind_seed$S.log 2>&1 &"
  sleep 45
done
echo "launched 12 runs (staggered), logs under $OUT/"
