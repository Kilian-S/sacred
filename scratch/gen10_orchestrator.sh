#!/bin/bash
# gen10_postfix orchestrator (pre-registered in experiments/gen10_postfix.md).
# Stage 1: multi-convoy headline re-run (3 seeds, 3-parallel).
# Stage 2: single-convoy B2-P3 re-run (3 seeds, 3-parallel).
# Stage 3: multi-convoy vanilla reference (1 seed).
# Detached-run pattern per SYSTEM.md (nohup + own session; harness-managed tasks got reaped once).
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen10_postfix
mkdir -p "$OUT"
PY=.venv/bin/python

echo "[gen10] stage 1: multi-convoy headline (3 seeds, 3-parallel)  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --seed $S --threads 3 --json-out "$OUT/mc_seed$S.json" \
    --ckpt-dir "$OUT/mc_seed${S}_ckpts" > "$OUT/mc_seed$S.log" 2>&1 &
done
wait
echo "[gen10] stage 1 done  $(date)" >> "$OUT/orchestrator.log"

echo "[gen10] stage 2: single-convoy B2-P3 (3 seeds, 3-parallel)  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_interdiction.py \
    --k-extra 8 --route-mode walk --attacker-mode smooth --sorties 3000 --seed $S \
    --eval-every 250 --threads 3 --json-out "$OUT/B2P3_seed$S.json" \
    > "$OUT/B2P3_seed$S.log" 2>&1 &
done
wait
echo "[gen10] stage 2 done  $(date)" >> "$OUT/orchestrator.log"

echo "[gen10] stage 3: multi-convoy vanilla reference (seed 0)  $(date)" >> "$OUT/orchestrator.log"
PYTHONPATH=. $PY scripts/train_multiconvoy.py \
  --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
  --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --sorties 1200 --eval-every 100 --seed 0 --threads 4 \
  --json-out "$OUT/van_seed0.json" > "$OUT/van_seed0.log" 2>&1
echo "[gen10] ALL DONE  $(date)" >> "$OUT/orchestrator.log"
touch "$OUT/DONE"
