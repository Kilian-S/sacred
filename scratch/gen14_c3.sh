#!/bin/bash
# gen14_evidence (C3) orchestrator: 10-seed both headline cells + the 35-159 vanilla row.
# Waves of 3 at 3 threads each. Pre-registered in experiments/gen14_evidence.md.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen14_evidence
mkdir -p "$OUT"
PY=.venv/bin/python
log () { echo "[gen14] $1  $(date)" >> "$OUT/orchestrator.log"; }

mc () {  # mc <seed>
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 35-159 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --seed "$1" --threads 3 --json-out "$OUT/mc_seed$1.json" \
    --ckpt-dir "$OUT/mc_seed$1_ckpts" > "$OUT/mc_seed$1.log" 2>&1
}
van () {
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 35-159 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --sorties 1200 --eval-every 100 --seed "$1" --threads 3 --vanilla-only \
    --json-out "$OUT/van_seed$1.json" > "$OUT/van_seed$1.log" 2>&1
}
sc () {
  PYTHONPATH=. $PY scripts/train_interdiction.py \
    --k-extra 8 --route-mode walk --attacker-mode smooth --sorties 3000 --seed "$1" \
    --eval-every 250 --threads 3 --json-out "$OUT/sc_seed$1.json" > "$OUT/sc_seed$1.log" 2>&1
}

log "stage 1: MC 35-159 seeds 0-9 (4 waves)"
for w in "0 1 2" "3 4 5" "6 7 8" "9"; do
  for S in $w; do mc "$S" & done
  wait
done
log "stage 2: 35-159 vanilla row seeds 0-2"
for S in 0 1 2; do van "$S" & done
wait
log "stage 3: SC 33-71 seeds 0-9 (4 waves; the long stage)"
for w in "0 1 2" "3 4 5" "6 7 8" "9"; do
  for S in $w; do sc "$S" & done
  wait
done
log "ALL DONE"
touch "$OUT/DONE"
