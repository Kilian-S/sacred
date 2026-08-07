#!/bin/bash
# gen41 ACT 3: chained launcher for WAVE 2 (seed 2 + head-only no-window control).
#
# Waits for the WAVE 1 processes to exit (polling explicit PIDs, never a pgrep pattern,
# which would match this script itself), then launches wave 2 into the SAME run directory
# so the four arms are evaluated together. Detached from any Claude session: once started
# with nohup ... & disown it survives terminal close, session end, and watcher reaping.
#
# Guards: refuses to double-launch if wave 2 artefacts already exist; verifies wave 2 at
# the POOL-LISTING + FIRST-TRAINING-PRINT level (the 2026-08-07 binding rule), never by
# process existence; logs every step with timestamps.
#
# Usage (already launched 2026-08-07):
#   nohup bash scratch/gen41_wave2_chain.sh > models/runs/gen41_act3w1/chain.log 2>&1 & disown

set -u
cd /Users/kilian/Kilian/ICL/Thesis/code/sacred

W1_PIDS="88754 88755"
RUN_DIR="models/runs/gen41_act3w1"
# NB: every flag is spelled out at both call sites below. Bundling them in a shell
# variable is deliberately avoided: that pattern silently failed to word-split earlier
# today and killed a launch.

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "chain armed; waiting on wave-1 PIDs: $W1_PIDS"

# ---- 1. wait for wave 1 -----------------------------------------------------------
while true; do
  alive=""
  for p in $W1_PIDS; do
    if kill -0 "$p" 2>/dev/null; then alive="$alive $p"; fi
  done
  [ -z "$alive" ] && break
  sleep 60
done
log "wave 1 processes have exited"

for S in 0 1; do
  if [ -f "$RUN_DIR/seed$S.json" ]; then
    log "  wave1 seed$S: JSON written -> $(grep -c 'sortie' "$RUN_DIR/seed$S.log") eval rounds"
    grep "select-on-train" "$RUN_DIR/seed$S.log" | tail -1 | sed 's/^/    /'
  else
    log "  wave1 seed$S: WARNING no JSON (crashed or was killed); wave 2 proceeds regardless"
    tail -3 "$RUN_DIR/seed$S.log" | sed 's/^/    /'
  fi
done

# ---- 2. guard against double launch (ATOMIC: mkdir succeeds for exactly one caller) --
# Redundant copies of this chain may be armed for resilience; the lock guarantees that
# exactly one of them ever launches wave 2, with no race window.
if [ -f "$RUN_DIR/seed2.json" ] || [ -f "$RUN_DIR/seed2.log" ]; then
  log "ABORT: wave-2 artefacts already exist; refusing to double-launch"
  exit 0
fi
if ! mkdir "$RUN_DIR/.wave2.lock" 2>/dev/null; then
  log "ABORT: another chain instance holds the wave-2 lock; standing down"
  exit 0
fi
log "acquired wave-2 launch lock"

# ---- 3. launch wave 2 --------------------------------------------------------------
log "launching wave 2: seed 2 + head-only no-window control"

OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. nohup \
  .venv/bin/python scripts/train_dyn_generalist.py \
  --pool-file models/runs/gen41_pool.json --K 2 --k-extra 12 --window 3 \
  --fast-refs --head-only --sorties 12000 --eval-every 500 --eval-n 600 \
  --eval-n-train 250 --threads 4 \
  --seed 2 --json-out "$RUN_DIR/seed2.json" --ckpt-dir "$RUN_DIR/seed2_ckpts" \
  > "$RUN_DIR/seed2.log" 2>&1 &
P2=$!

OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. nohup \
  .venv/bin/python scripts/train_dyn_generalist.py \
  --pool-file models/runs/gen41_pool.json --K 2 --k-extra 12 --window 3 \
  --fast-refs --head-only --sorties 12000 --eval-every 500 --eval-n 600 \
  --eval-n-train 250 --threads 4 \
  --no-window --seed 0 --json-out "$RUN_DIR/seed0_nowin.json" \
  --ckpt-dir "$RUN_DIR/seed0_nowin_ckpts" \
  > "$RUN_DIR/seed0_nowin.log" 2>&1 &
PC=$!

log "wave 2 PIDs: seed2=$P2 control=$PC"

# ---- 4. verify at pool-listing + first-print level (never by process existence) -----
deadline=$(( $(date +%s) + 3600 ))
ok2=0; okc=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if [ $ok2 -eq 0 ] && grep -q "sortie" "$RUN_DIR/seed2.log" 2>/dev/null; then
    ok2=1; log "VERIFIED seed2 training: $(grep 'sortie' "$RUN_DIR/seed2.log" | tail -1)"
  fi
  if [ $okc -eq 0 ] && grep -q "sortie" "$RUN_DIR/seed0_nowin.log" 2>/dev/null; then
    okc=1; log "VERIFIED control training: $(grep 'sortie' "$RUN_DIR/seed0_nowin.log" | tail -1)"
  fi
  [ $ok2 -eq 1 ] && [ $okc -eq 1 ] && break
  if ! kill -0 "$P2" 2>/dev/null && [ $ok2 -eq 0 ]; then
    log "FAILURE seed2 died before its first training print:"; tail -15 "$RUN_DIR/seed2.log" | sed 's/^/    /'
  fi
  if ! kill -0 "$PC" 2>/dev/null && [ $okc -eq 0 ]; then
    log "FAILURE control died before its first training print:"; tail -15 "$RUN_DIR/seed0_nowin.log" | sed 's/^/    /'
  fi
  sleep 30
done

if [ $ok2 -eq 1 ] && [ $okc -eq 1 ]; then
  log "WAVE 2 LAUNCHED AND VERIFIED (both arms training)"
else
  log "WAVE 2 VERIFICATION INCOMPLETE (seed2=$ok2 control=$okc) - see logs above"
fi

# ---- 5. record completion for the morning ------------------------------------------
while kill -0 "$P2" 2>/dev/null || kill -0 "$PC" 2>/dev/null; do sleep 120; done
log "wave 2 processes have exited"
for A in seed2 seed0_nowin; do
  if [ -f "$RUN_DIR/$A.json" ]; then
    log "  $A: COMPLETE"; grep "select-on-train" "$RUN_DIR/$A.log" | tail -1 | sed 's/^/    /'
  else
    log "  $A: NO JSON (crashed/killed)"; tail -3 "$RUN_DIR/$A.log" | sed 's/^/    /'
  fi
done
log "ALL FOUR ARMS DONE - ready for the high-precision pass and the three-tier verdict"
