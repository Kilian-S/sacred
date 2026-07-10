#!/bin/bash
# ZST step 0 (night programme item 5): waits for the gen12 sweeps, then retrains the post-fix
# single-convoy source policy with actor saving (gen10-SC config, seed 0, sacred arm only), then
# runs the held-out-OD transfer eval (scratch/zst_transfer.py -> 110-135 k8 = the unrun B2-S).
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/zst_step0
mkdir -p "$OUT"
PY=.venv/bin/python
until [ -f models/runs/gen12_sweeps/DONE ]; do sleep 60; done
echo "[zst] source retrain start  $(date)" >> "$OUT/stage.log"
PYTHONPATH=. $PY scripts/train_interdiction.py \
  --k-extra 8 --route-mode walk --attacker-mode smooth --sorties 3000 --seed 0 \
  --eval-every 250 --threads 4 --skip-vanilla \
  --save-actor "$OUT/source_actor_3371.pt" \
  --json-out "$OUT/source_retrain.json" > "$OUT/source_retrain.log" 2>&1
echo "[zst] transfer eval  $(date)" >> "$OUT/stage.log"
PYTHONPATH=. $PY scratch/zst_transfer.py "$OUT/source_actor_3371.pt" > "$OUT/transfer.log" 2>&1
echo "[zst] DONE  $(date)" >> "$OUT/stage.log"
touch "$OUT/DONE"
