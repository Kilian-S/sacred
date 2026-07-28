#!/bin/bash
# Standalone progress log for the gen39 step-5 batch (session-independent: nohup'd, reparented to
# launchd). Reads models/runs/gen39_step5/*.log and appends every new checkpoint line.
O="$(cd "$(dirname "$0")/.." && pwd)/models/runs/gen39_step5"
LOG="$O/../gen39_step5_night.log"
TOTAL=${1:-12}
n=0
while true; do
  alive=$(pgrep -f 'Python.*train_gen39_conc[e]al' | wc -l | tr -d ' ')
  done_json=$(ls "$O"/*_seed*.json 2>/dev/null | wc -l | tr -d ' ')
  m=$(grep -h "sortie" "$O"/*.log 2>/dev/null | wc -l | tr -d ' ')
  if [ "$m" -gt "$n" ]; then
    echo "[$(date '+%m-%d %H:%M')] --- new checkpoints (alive=$alive, finished=$done_json/$TOTAL) ---" >> "$LOG"
    grep -H "sortie" "$O"/*.log 2>/dev/null | sed 's|.*/||;s|\.log:|  |' | tail -n $((m-n)) >> "$LOG"
    n=$m
  fi
  if [ "$alive" -eq 0 ] && [ "$done_json" -ge "$TOTAL" ]; then
    echo "[$(date '+%m-%d %H:%M')] ALL $TOTAL RUNS COMPLETE" >> "$LOG"; exit 0
  fi
  sleep 300
done
