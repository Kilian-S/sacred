#!/usr/bin/env bash
# gen09-STAB: leader-stabilisation re-run (fleet-route 62->97 k8, N=3, K=1, smooth FP).
# Kills the across-seed leader-alpha-collapse variance via a leader-alpha floor + a notch higher
# leader-ent-frac + steadier/longer smooth FP. Seeds 0/1/2, 3-parallel (3 threads each = 9 <= 10
# cores). ALL outputs saved (JSON + tee'd log) -> the first reproducible fleet-route number.
# Pre-registered in experiments/gen09_multiconvoy.md (gen09-STAB). Run from the repo root:
#   bash scratch/gen09_leader_stab.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=models/runs/gen09_multiconvoy
mkdir -p "$OUT"
run_seed() {
  local S=$1
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.15 --switch-every 200 \
    --leader-ent-frac 0.6 --leader-alpha-floor 0.3 --sorties 1200 --eval-every 200 \
    --seed "$S" --threads 3 \
    --json-out "$OUT/fleetroute_stab_seed$S.json" \
    > "$OUT/fleetroute_stab_seed$S.log" 2>&1
}
for S in 0 1 2; do run_seed "$S" & done
wait
echo "gen09-STAB done. Per-seed leader TAP:"
for S in 0 1 2; do
  printf "  seed %s: " "$S"
  PYTHONPATH=. .venv/bin/python -c "import json,sys; d=json.load(open('$OUT/fleetroute_stab_seed$S.json')); f=d['fleet_route']; print('TAP', round(f['expl_tap'],3), '| policy', round(f['expl'],3), '| stack', round(f['stack_rate'],2))"
done
