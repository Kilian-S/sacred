#!/bin/zsh
# gen41 Act-2 completion watcher (armed 2026-08-06 on Kilian's go): waits until all four
# arms have written their result JSONs, then fires the pre-registered final evaluation
# (scratch/gen41_final_eval_act2.py). Aborts loudly if the trainers vanish without JSONs.
cd "$(dirname "$0")/.."
F=(models/runs/gen41_act2/seed0.json models/runs/gen41_act2/seed1.json models/runs/gen41_act2/seed2.json models/runs/gen41_act2/seed0_nowin.json)
while :; do
  n=0; for f in $F; do [ -f $f ] && n=$((n+1)); done
  [ $n -eq 4 ] && break
  if ! pgrep -f "train_dyn_generalist.*gen41_act2" >/dev/null; then
    echo "TRAINERS GONE with $n/4 JSONs - aborting watcher $(date)"; exit 1
  fi
  sleep 120
done
echo "BATCH COMPLETE $(date) - firing final evaluation"
sleep 30
OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. .venv/bin/python scratch/gen41_final_eval_act2.py
echo "FINAL EVAL DONE $(date)"
