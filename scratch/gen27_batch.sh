#!/bin/bash
# gen27 batch: 3 history-aware seeds at 3-parallel, then the no-window causal control.
# Ledger: experiments/gen27_dynamic_generalist.md
set -u
cd "$(dirname "$0")/.."
mkdir -p models/runs/gen27_dyn_generalist
for S in 0 1 2; do
  PYTHONPATH=. .venv/bin/python scripts/train_dyn_generalist.py \
    --sorties 12000 --eval-every 500 --seed $S --threads 3 \
    --json-out models/runs/gen27_dyn_generalist/seed$S.json \
    --ckpt-dir models/runs/gen27_dyn_generalist/seed${S}_ckpts \
    > models/runs/gen27_dyn_generalist/seed$S.log 2>&1 &
done
wait
PYTHONPATH=. .venv/bin/python scripts/train_dyn_generalist.py \
  --sorties 12000 --eval-every 500 --seed 0 --threads 4 --no-window \
  --json-out models/runs/gen27_dyn_generalist/seed0_nowin.json \
  --ckpt-dir models/runs/gen27_dyn_generalist/seed0_nowin_ckpts \
  > models/runs/gen27_dyn_generalist/seed0_nowin.log 2>&1
echo "GEN27_BATCH_DONE"
