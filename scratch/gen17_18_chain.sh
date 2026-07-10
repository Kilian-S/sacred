#!/bin/bash
# C4 (gen17 annealed-smoothing last-iterate attempt) then C2 (gen18 learned-follower redo),
# chained behind gen16. Pre-registered in experiments/gen17_lastiterate.md / gen18_learnedfollower.md.
set -u
cd "$(dirname "$0")/.."
PY=.venv/bin/python
G17=models/runs/gen17_lastiterate; G18=models/runs/gen18_learnedfollower
mkdir -p "$G17" "$G18"
until [ -f models/runs/gen16_multicity/DONE ]; do sleep 180; done

echo "[gen17] start  $(date)" >> "$G17/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 35-159 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --fp-tau-final 0.02 \
    --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 2400 --eval-every 100 \
    --seed $S --threads 3 --json-out "$G17/seed$S.json" \
    --ckpt-dir "$G17/seed${S}_ckpts" > "$G17/seed$S.log" 2>&1 &
done
wait
echo "[gen17] done  $(date)" >> "$G17/orchestrator.log"
touch "$G17/DONE"

echo "[gen18] start  $(date)" >> "$G18/orchestrator.log"
for S in 0 1 2; do
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od 35-159 --N 3 --K 1 --k-extra 8 --menu-select --band 0.15,0.95 \
    --attacker-mode smooth --fp-tau 0.15 --switch-every 200 --smooth-window 250 \
    --leader-ckpt models/runs/gen14_evidence/mc_seed1_ckpts/actor_ep500.pt \
    --forced-copy-warmup 600 --stack-dup 4 --head-term-lr 3e-2 --skip-vanilla \
    --sorties 3200 --eval-every 200 \
    --seed $S --threads 3 --json-out "$G18/seed$S.json" > "$G18/seed$S.log" 2>&1 &
done
wait
echo "[gen18] done  $(date)" >> "$G18/orchestrator.log"
touch "$G18/DONE"
