#!/bin/bash
# gen37_reasoning_curation training batch (pre-registered: experiments/gen37_reasoning_curation.md;
# Kilian's full launch control 2026-07-24). 3 arms x 3 seeds, staged 6-then-3 to bound
# oversubscription; identical trainer/budget, only the shortlist differs.
# Resume-safe-ish: each run writes its own --json-out at the end; a kill loses only in-flight runs.
set -u
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/kilian/Kilian/ICL/Thesis/code/sacred/.venv/bin/python
OUT=models/runs/gen37_reasoning_curation
mkdir -p "$OUT"
echo "[gen37] start $(date) sha=$(git rev-parse --short HEAD)"

run() {  # arm seed
  local arm=$1 s=$2
  PYTHONPATH=. $PY scripts/train_multiod_generalist.py \
    --shortlist $OUT/shortlists_${arm}.json --sorties 14000 --eval-every 1000 --seed $s \
    --threads 2 --json-out $OUT/${arm}_s${s}.json --ckpt-dir $OUT/${arm}_s${s}_ckpts \
    > $OUT/${arm}_s${s}.log 2>&1
}

# wave 1: 6 runs (llm+random all seeds)
for arm in llm random; do for s in 0 1 2; do run $arm $s & done; done
wait
echo "[gen37] wave 1 (llm,random) done $(date)"
# wave 2: 3 runs (heuristic)
for s in 0 1 2; do run heuristic $s & done
wait
echo "[gen37] wave 2 (heuristic) done $(date)"
echo "[gen37] ALL DONE $(date)"
