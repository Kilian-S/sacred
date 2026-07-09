#!/usr/bin/env bash
# gen09 HEADLINE: the definitive, locked multi-convoy fleet-route run (62->97 k8, N=3, K=1).
# Kilian's decision (2026-07-09): stop the STAB stabilisation chase; the leader's late drift is
# inherent last-iterate FP cycling, resolved the standard single-convoy way = BEST-CHECKPOINT
# selection (the final iterate over-trains toward uniform). This run uses the sharp-attacker config
# (tau 0.05) that reproducibly reaches the equilibrium hedge, on the current (true-smooth fp_dynamics)
# code, with FULL SAVING (json + per-eval actor checkpoints) so the best-checkpoint is a real
# re-evaluable artefact. ~1200 sorties so the plot shows BOTH the best-checkpoint AND the drift.
# 3 seeds, 3-parallel. Best-checkpoint per seed = lowest exploitability (TAP). Run from repo root:
#   bash scratch/gen09_headline.sh
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=models/runs/gen09_multiconvoy
mkdir -p "$OUT"
run_seed() {
  local S=$1
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --seed "$S" --threads 3 \
    --json-out "$OUT/headline_seed$S.json" --ckpt-dir "$OUT/headline_seed${S}_ckpts" \
    > "$OUT/headline_seed$S.log" 2>&1
}
for S in 0 1 2; do run_seed "$S" & done
wait
echo "=== gen09 HEADLINE done: best-checkpoint ladder ==="
PYTHONPATH=. .venv/bin/python -c "
import json, statistics
bt=[]; be=[]
for s in (0,1,2):
    d=json.load(open('$OUT/headline_seed%d.json'%s))['fleet_route']
    bt.append(d['best_tap']); be.append(d['best_expl'])
    print('  seed %d: best-ckpt TAP %.3f @ sortie %s | single-ckpt %.3f @ %s | final TAP %.3f (drift)'
          % (s, d['best_tap'], d['best_tap_sortie'], d['best_expl'], d['best_expl_sortie'], d['expl_tap']))
b=json.load(open('$OUT/headline_seed0.json'))
print('  SACRED best-ckpt TAP mean %.3f +/- %.3f (3 seeds)' % (statistics.mean(bt), statistics.pstdev(bt)))
print('  loss_mixed (equilibrium) %.3f' % b['loss_mixed'])
"
PYTHONPATH=. .venv/bin/python -c "
from src.envs.multiconvoy_interdiction import make_multiconvoy_env
from src.baselines.multiconvoy_planners import classical_baselines
e=make_multiconvoy_env(od=('62','97'),N=3,K=1,edge_vuln_band=(0.15,0.95),k_extra_routes=8,menu_select=True)
b=classical_baselines(e.game,3,'mission')
print('  LADDER: shortest %.3f > vanilla ~0.945 > ALNS %.3f > SACRED(best-ckpt) [above] > equilibrium %.3f'
      % (b['shortest_path'], b['alns'], b['equilibrium']))
print('  FAIRNESS: ALNS spread %.3f vs ALNS_forced_stack %.3f (ALNS spreads by CHOICE; SACRED beats spread by RANDOMISING)'
      % (b['alns'], b['alns_forced_stack']))
"
