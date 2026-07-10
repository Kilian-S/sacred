#!/bin/bash
# Post-A1 eval chain: waits for gen15 (A1), picks the best seed's best-checkpoint actor, then runs
# A3 (amortisation) and A2 (graph transfer). EVAL-ONLY. Results into their ledgers by the summariser.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
OUT=models/runs; mkdir -p "$OUT"
until [ -f models/runs/gen15_generalist/DONE ]; do sleep 180; done
echo "[post-a1] gen15 done; selecting best actor  $(date)" >> "$OUT/post_a1.log"

# best seed = lowest best_test_ratio; best checkpoint actor within it (by the run's best_at)
ACTOR=$($PY - <<'PY'
import json, glob, re
best=(1e9,None,None)
for f in glob.glob("models/runs/gen15_generalist/seed*.json"):
    d=json.load(open(f)); s=re.search(r"seed(\d+)",f).group(1)
    if d.get("best_test_ratio",1e9)<best[0]:
        best=(d["best_test_ratio"], s, d.get("best_at"))
r,s,at=best
cks=f"models/runs/gen15_generalist/seed{s}_ckpts/actor_ep{at}.pt"
import os
print(cks if os.path.exists(cks) else f"models/runs/gen15_generalist/seed{s}_ckpts/"+sorted(os.listdir(f"models/runs/gen15_generalist/seed{s}_ckpts"))[-1])
PY
)
echo "[post-a1] best actor = $ACTOR  $(date)" >> "$OUT/post_a1.log"

# A1 training wall (max seed wall from the logs), for A3's crossover line
TC=$($PY - <<'PY'
import glob, re
best=0
for f in glob.glob("models/runs/gen15_generalist/seed*.log"):
    for line in open(f):
        m=re.search(r"\|\s+(\d+)s\s*$", line.strip())
        if m: best=max(best,int(m.group(1)))
print(best or 6000)
PY
)

echo "[post-a1] A3 amortisation  $(date)" >> "$OUT/post_a1.log"
PYTHONPATH=. $PY scratch/amortisation_benchmark.py "$ACTOR" --train-cost-s "$TC" \
  > "$OUT/a3_amortisation.log" 2>&1
echo "[post-a1] A2 graph transfer  $(date)" >> "$OUT/post_a1.log"
PYTHONPATH=. $PY scratch/a2_graph_transfer.py "$ACTOR" --tag original \
  > "$OUT/a2_graph_transfer.log" 2>&1
echo "[post-a1] DONE  $(date)" >> "$OUT/post_a1.log"
touch "$OUT/post_a1_DONE"
