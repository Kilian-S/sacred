#!/bin/bash
# gen45 NIGHT CHAIN (2026-08-09 23:0x). Kilian's instruction: the gen45 attempt wave must fire
# the moment the gen39 step-5e wave clears, and the Mac must not idle overnight.
#
# Waits on the three named 5e trainer PIDs AND on the absence of any train_gen39_conceal.py
# process (belt and braces against PID reuse), settles, then runs the pinned attempt wave via
# scratch/gen45_batch.sh. Heartbeats hourly so the analyst can follow progress without tailing.
# The confirmation wave is deliberately NOT chained: it evaluates the pristine gated set ONCE,
# so it fires only after the attempt diagnostics have been read (pre-registration, gen32
# precedent).
cd "$(dirname "$0")/.."
OUT=models/runs/gen45_unified
mkdir -p "$OUT"
PIDS="27673 27676 27678"
say() { echo "[CHAIN $(date '+%d %H:%M:%S')] $*"; }

say "armed at nice=$(ps -o nice= -p $$ | tr -d ' '); waiting on 5e PIDs $PIDS"
DEADLINE=$(( $(date +%s) + 8 * 3600 ))
while :; do
  alive=0
  for p in $PIDS; do kill -0 "$p" 2>/dev/null && alive=$((alive + 1)); done
  rem=$(pgrep -f train_gen39_conceal.py | wc -l | tr -d ' ')
  if [ "$alive" -eq 0 ] && [ "$rem" -eq 0 ]; then
    say "5e CLEAR (named PIDs gone, no conceal trainers left)"
    break
  fi
  if [ "$(date +%s)" -gt "$DEADLINE" ]; then
    say "WARNING deadline hit with alive=$alive rem=$rem; firing anyway"
    break
  fi
  sleep 60
done

sleep 60                      # let the 5e runs flush their state files and release memory
say "launching gen45 ATTEMPT wave: 3 runs, seeds 0/1/2, 16000 sorties, THREADS=2"
THREADS=2 bash scratch/gen45_batch.sh attempt &
WAVE=$!
sleep 120
say "attempt PIDs/nice: $(ps -ax -o pid,nice,command | grep train_gen45_unified | grep -v grep \
  | awk '{printf "%s(ni%s) ", $1, $2}')"

while kill -0 "$WAVE" 2>/dev/null; do
  sleep 3600
  kill -0 "$WAVE" 2>/dev/null || break
  say "heartbeat"
  for s in 0 1 2; do
    tail -1 "$OUT/attempt_seed$s.log" 2>/dev/null | sed "s/^/[CHAIN   seed$s] /"
  done
done
say "ATTEMPT WAVE COMPLETE (json: $(ls $OUT/attempt_seed*.json 2>/dev/null | wc -l | tr -d ' ')/3)"
say "STOP. The confirmation wave needs the attempt diagnostics read first (pre-registration)."
