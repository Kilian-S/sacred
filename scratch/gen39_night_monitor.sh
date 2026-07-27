#!/bin/bash
# Standalone progress log for the gen39 step-5 batch (session-independent; nohup'd, reparented).
O="$(cd "$(dirname "$0")/.." && pwd)/models/runs/gen39_step5"
LOG="$O/../gen39_step5_night.log"
n=0
while true; do
  alive=$(pgrep -f 'Python.*train_gen39_conc[e]al' | wc -l | tr -d ' ')
  done_json=$(ls "$O"/*.json 2>/dev/null | grep -v curricula | wc -l | tr -d ' ')
  m=$(grep -h "sortie" "$O"/*.log 2>/dev/null | wc -l | tr -d ' ')
  if [ "$m" -gt "$n" ]; then
    echo "[$(date '+%m-%d %H:%M')] --- new checkpoints (alive=$alive, finished=$done_json/8) ---" >> "$LOG"
    grep -H "sortie" "$O"/*.log 2>/dev/null | sed 's|.*/||;s|\.log:|  |' | tail -n $((m-n)) >> "$LOG"
    n=$m
  fi
  if [ "$alive" -eq 0 ] && [ "$done_json" -ge 8 ]; then
    echo "[$(date '+%m-%d %H:%M')] ALL 8 RUNS COMPLETE" >> "$LOG"; exit 0
  fi
  sleep 300
done
