#!/bin/bash
# Morning steps 1-3 (Kilian's go, 2026-07-10): gen13-lock -> gen11b -> oracle-scaling re-measure.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=models/runs/gen13_lock; mkdir -p "$L"
G=models/runs/gen11_menuhead
log () { echo "[morning] $1  $(date)" >> "$L/orchestrator.log"; }

log "stage 1: gen13-lock 35-159 N3K1 x 3 seeds"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 35-159 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --seed $S --threads 3 --json-out "$L/seed$S.json" \
    --ckpt-dir "$L/seed${S}_ckpts" > "$L/seed$S.log" 2>&1 &
done
wait
log "stage 1 done"

log "stage 2: gen11b arms B' (features) and E' (identity), head-term-lr 3e-2"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --route-feats --head-term-lr 3e-2 --seed $S --threads 3 \
    --json-out "$G/Bp_seed$S.json" --ckpt-dir "$G/Bp_seed${S}_ckpts" > "$G/Bp_seed$S.log" 2>&1 &
done
wait
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 62-97 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --route-bias --head-term-lr 3e-2 --seed $S --threads 3 \
    --json-out "$G/Ep_seed$S.json" --ckpt-dir "$G/Ep_seed${S}_ckpts" > "$G/Ep_seed$S.log" 2>&1 &
done
wait
log "stage 2 done"

log "stage 3: oracle-scaling probe re-measure (uncontended)"
PYTHONPATH=. $PY scratch/oracle_scaling_probe.py > scratch/oracle_scaling_output_v2.txt 2>&1
log "ALL DONE"
touch "$L/DONE"
