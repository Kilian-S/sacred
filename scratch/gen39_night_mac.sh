#!/bin/zsh
# Mac night driver (2026-08-08): keeps the Mac working continuously through the step-5c and
# gen43 arm. Each stage waits only for its own prerequisite, so nothing sits idle. The GPU box
# is driven separately by the stagehand; this script never touches it.
#
# Stages
#   1  wait for the llm16 fresh-test-family build (box-dependent, already chained)
#   2  score the 12 BANKED checkpoints on the fresh sets, one process per map (parallel)
#   3  wait for the three qwenthink16 training runs to finish
#   4  mark step-5c against its pre-registered clauses (narva, test set unchanged)
#   5  score the 3 NEW checkpoints on the fresh sets (same per-map parallelism), then merge
#   6  final marking sweep (step-5c + whatever gen43 papers exist by then)
set -u
cd "$(dirname "$0")/.."
PY=../sacred/.venv/bin/python
R=models/runs/gen39_step5
MAPS=(kgd_gvardeysk ukraine fulda)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

say() { echo "=== $1  $(date '+%H:%M:%S') ==="; }

say "STAGE 1 waiting for the llm16 test-family build"
while pgrep -f "zeroshot2.py --build" >/dev/null; do sleep 30; done
say "STAGE 1 done"

say "STAGE 2 scoring the banked checkpoints, one process per map"
for M in $MAPS; do
  PYTHONPATH=. nohup $PY scratch/gen39_zeroshot2.py --score --maps $M \
      --out $R/zeroshot2_score_$M.json > $R/zeroshot2_score_$M.log 2>&1 &
done
wait
say "STAGE 2 done"

say "STAGE 3 waiting for the qwenthink16 training runs"
while pgrep -f "train_gen39_conceal.py --arm qwenthink16" >/dev/null; do sleep 60; done
say "STAGE 3 done"

say "STAGE 4 step-5c marking (narva)"
PYTHONPATH=. $PY scratch/gen39_step5c_score.py > $R/step5c_verdict.txt 2>&1
cat $R/step5c_verdict.txt

say "STAGE 5 scoring the new checkpoints on the fresh sets"
for M in $MAPS; do
  PYTHONPATH=. nohup $PY scratch/gen39_zeroshot2.py --score --maps $M \
      --out $R/zeroshot2_score_$M.json >> $R/zeroshot2_score_$M.log 2>&1 &
done
wait
PYTHONPATH=. $PY scratch/gen39_zeroshot2.py --merge
say "STAGE 5 done"

say "STAGE 6 final marking sweep"
PYTHONPATH=. $PY scratch/gen39_step5c_score.py > $R/step5c_verdict.txt 2>&1
PYTHONPATH=. $PY scratch/gen43_mark.py > models/runs/gen43_exam/marks.txt 2>&1
tail -25 models/runs/gen43_exam/marks.txt
say "MAC NIGHT DRIVER COMPLETE"
