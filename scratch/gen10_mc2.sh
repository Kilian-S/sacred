#!/bin/bash
# gen10-MC2 diagnostic (pre-registered in experiments/gen10_postfix.md, Kilian's go 2026-07-10):
# multi-convoy headline config at 2400 sorties with --legacy-role-target, 3 seeds, 3-parallel.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen10_postfix
mkdir -p "$OUT"
PY=.venv/bin/python
echo "[gen10-MC2] start  $(date)" >> "$OUT/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 2400 --eval-every 100 \
    --legacy-role-target --seed $S --threads 3 \
    --json-out "$OUT/mc2_seed$S.json" \
    --ckpt-dir "$OUT/mc2_seed${S}_ckpts" > "$OUT/mc2_seed$S.log" 2>&1 &
done
wait
echo "[gen10-MC2] done  $(date)" >> "$OUT/orchestrator.log"
touch "$OUT/MC2_DONE"
