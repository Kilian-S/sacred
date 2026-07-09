#!/usr/bin/env bash
# gen09-STAB-2: corrected leader-stabilisation re-run (fleet-route 62->97 k8, N=3, K=1).
# STAB-1 failed: fp-tau 0.15 (borrowed from the FOLLOWER attempt 6) was too diffuse for the LEADER,
# so the leader had no per-route Q-gradient and drifted to UNIFORM (TAP ~0.77, worse than ALNS).
# Fix: SHARP attacker fp-tau 0.05 (the leader's setting) = the adversarial driver that forces the
# ~1/vuln hedge; floor lowered 0.30 -> 0.20 (0.30 was CLAMPING all seeds); leader-ent-frac 0.6 -> 0.5
# (BELOW the 0.63 equilibrium, so landing at ~0.63 must be the attacker's doing, not the target).
# The floor + ent-frac are PERMISSIVE GUARDRAILS, NOT tuned to the oracle answer (Kilian 2026-07-09).
# Seeds 0/1/2, 3-parallel, all outputs saved. Run from repo root: bash scratch/gen09_leader_stab2.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=models/runs/gen09_multiconvoy
mkdir -p "$OUT"
run_seed() {
  local S=$1
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 200 \
    --seed "$S" --threads 3 \
    --json-out "$OUT/fleetroute_stab2_seed$S.json" \
    > "$OUT/fleetroute_stab2_seed$S.log" 2>&1
}
for S in 0 1 2; do run_seed "$S" & done
wait
echo "gen09-STAB-2 done. Per-seed leader TAP (target: land near eq 0.216, TAP ~0.25-0.30 tight):"
for S in 0 1 2; do
  printf "  seed %s: " "$S"
  PYTHONPATH=. .venv/bin/python -c "import json; d=json.load(open('$OUT/fleetroute_stab2_seed$S.json'))['fleet_route']; print('TAP', round(d['expl_tap'],3), '| policy', round(d['expl'],3), '| H_lead/lnR', round(d['H_lead']/2.4849,2), '| stack', round(d['stack_rate'],2))"
done
