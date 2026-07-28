#!/bin/bash
# Detached sequencer: wait for gen37 to finish, then launch gen38 V2 (avoids CPU oversubscription).
set -u
while ! grep -q "ALL DONE" "/Users/kilian/Kilian/ICL/Thesis/code/sacred-gen29/models/runs/gen37_reasoning_curation/orchestrator.log" 2>/dev/null; do
  pgrep -f train_multiod_generalist >/dev/null 2>&1 || break   # gen37 gone (done or died)
  sleep 120
done
echo "[seq] gen37 clear at $(date); launching V2" >> /Users/kilian/Kilian/ICL/Thesis/code/sacred/models/runs/gen38_llm_enemy_id/sequencer.log
bash /Users/kilian/Kilian/ICL/Thesis/code/sacred/scratch/gen38_v2_batch.sh >> /Users/kilian/Kilian/ICL/Thesis/code/sacred/models/runs/gen38_llm_enemy_id/sequencer.log 2>&1
