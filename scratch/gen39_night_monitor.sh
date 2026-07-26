#!/bin/bash
# Standalone overnight progress log for the gen39 step-3 batch. Independent of any agent session:
# nohup'd, reparented to launchd, appends a timestamped line whenever a run reports a checkpoint
# or a wave changes. Read models/runs/gen39_step3_night.log in the morning.
O="$(cd "$(dirname "$0")/.." && pwd)/models/runs/gen39_step3"
LOG="$O/../gen39_step3_night.log"
n=0
while true; do
  alive=$(pgrep -f 'Python.*train_gen39_conc[e]al' | wc -l | tr -d ' ')
  done_json=$(ls "$O"/*.json 2>/dev/null | grep -v pool_cache | wc -l | tr -d ' ')
  m=$(grep -h "sortie" "$O"/*.log 2>/dev/null | wc -l | tr -d ' ')
  if [ "$m" -gt "$n" ]; then
    echo "[$(date '+%m-%d %H:%M')] --- new checkpoints (alive=$alive, finished=$done_json/12) ---" >> "$LOG"
    grep -H "sortie" "$O"/*.log 2>/dev/null | sed 's|.*/||;s|\.log:|  |' | tail -n $((m-n)) >> "$LOG"
    n=$m
  fi
  if [ "$alive" -eq 0 ] && [ "$done_json" -ge 12 ]; then
    echo "[$(date '+%m-%d %H:%M')] ALL 12 RUNS COMPLETE" >> "$LOG"; exit 0
  fi
  if [ "$alive" -eq 0 ]; then
    echo "[$(date '+%m-%d %H:%M')] WARNING: no trainers alive, only $done_json/12 finished" >> "$LOG"
    sleep 300
  fi
  sleep 300
done
