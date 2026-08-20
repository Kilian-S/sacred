#!/bin/zsh
# gen39 phases 1c/1d/1e on the corrected v2-table brief, plus the 1e thinking-on rider.
# Existing artefacts are copied aside as *_v1brief.* before anything runs.
set -e
cd "$(dirname "$0")/.."
PY=../sacred/.venv/bin/python
R=models/runs

for f in gen39_phase1c gen39_phase1d gen39_phase1e gen39_phase1e_thinking gen39_phase1e_thinking_traces; do
  if [ -f $R/$f.json ]; then cp -n $R/$f.json $R/${f}_v1brief.json || true; fi
done
for f in forces_robust.json forces_iter.json brief_robust.txt brief_phase1d.txt brief_phase1e.txt; do
  if [ -f $R/gen39_compose/$f ]; then cp -n $R/gen39_compose/$f $R/gen39_compose/${f%.*}_v1brief.${f##*.} || true; fi
done

export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

echo "=== 1c START $(date) ==="
PYTHONPATH=. $PY analysis/gen39_phase1c.py --robust --iter --curated
echo "=== 1d START $(date) ==="
PYTHONPATH=. $PY analysis/gen39_phase1d.py --rounds 6 --n 3
echo "=== 1e START $(date) ==="
PYTHONPATH=. $PY analysis/gen39_phase1e.py --n 4 --rounds 2
echo "=== 1e THINKING RIDER START $(date) ==="
PYTHONPATH=. $PY analysis/gen39_phase1e_thinking.py
echo "=== ALL DONE $(date) ==="
