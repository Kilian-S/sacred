#!/usr/bin/env bash
# gen09-STAB-3: leader-stabilisation with the TRUE smooth-FP attacker (fleet-route 62->97 k8, N=3, K=1).
# STAB-2 proved the sharp attacker (tau 0.05) drives the leader to the equilibrium hedge EARLY
# (TAP 0.27-0.29), but the multi-convoy "smooth" attacker block-held one iset for the whole
# switch_every and averaged all-history, so the tail FP-cycled away. STAB-3 uses the PORTED, shared,
# B2-P3-proven smooth-FP discipline (src/baselines/fp_dynamics.py, now used by both trainers):
# softmax BR to a TRAILING-WINDOW of recent occupancy, with a FRESH iset sampled EVERY sortie.
# ONLY change vs STAB-2 = the attacker mechanism (a MECHANISM port, NOT a knob tune): tau 0.05, floor
# 0.20, ent-frac 0.5, switch 200 all unchanged; smooth-window 250. Seeds 0/1/2, 3-parallel, all saved.
# Success (Kilian): leader HOLDS ~0.27 across the tail (not diverging to 0.6-0.8), per-eval spikes damp
# (judge the trailing TAP), seed-to-seed variance gone. Run from repo root: bash scratch/gen09_leader_stab3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=models/runs/gen09_multiconvoy
mkdir -p "$OUT"
run_seed() {
  local S=$1
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 200 \
    --seed "$S" --threads 3 \
    --json-out "$OUT/fleetroute_stab3_seed$S.json" \
    > "$OUT/fleetroute_stab3_seed$S.log" 2>&1
}
for S in 0 1 2; do run_seed "$S" & done
wait
echo "gen09-STAB-3 done. Per-seed leader TAP (want tight ~0.27, held across the tail):"
for S in 0 1 2; do
  printf "  seed %s: " "$S"
  PYTHONPATH=. .venv/bin/python -c "import json; d=json.load(open('$OUT/fleetroute_stab3_seed$S.json'))['fleet_route']; print('TAP', round(d['expl_tap'],3), '| tail_expl', round(d.get('tail_expl',float('nan')),3), '| tail_amp', round(d.get('tail_amp',float('nan')),3), '| H_lead/lnR', round(d['H_lead']/2.4849,2))"
done
