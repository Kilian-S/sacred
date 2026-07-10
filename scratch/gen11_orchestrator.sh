#!/bin/bash
# gen11_menuhead orchestrator (pre-registered in experiments/gen11_menuhead.md).
# Arms B (features), C (leader-only push), D (both), E (identity bias); 3 seeds each,
# 3-parallel within an arm, arms serial. Detached-run pattern per SYSTEM.md.
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen11_menuhead
mkdir -p "$OUT"
PY=.venv/bin/python

run_arm () {
  local name=$1; shift
  echo "[gen11] arm $name start  $(date)" >> "$OUT/orchestrator.log"
  for S in 0 1 2; do
    PYTHONPATH=. $PY scripts/train_multiconvoy.py \
      --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
      --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
      --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
      "$@" --seed $S --threads 3 \
      --json-out "$OUT/${name}_seed$S.json" \
      --ckpt-dir "$OUT/${name}_seed${S}_ckpts" > "$OUT/${name}_seed$S.log" 2>&1 &
  done
  wait
  echo "[gen11] arm $name done  $(date)" >> "$OUT/orchestrator.log"
}

run_arm B --route-feats
run_arm C --leader-only-push
run_arm D --route-feats --leader-only-push
run_arm E --route-bias
echo "[gen11] ALL DONE  $(date)" >> "$OUT/orchestrator.log"
touch "$OUT/DONE"
