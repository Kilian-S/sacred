#!/bin/bash
# gen26 step 1: n=3 the K=3 crossover cell on 35-159 (ledger: experiments/gen26_kboundary.md)
# 3 seeds at 3-parallel, detached; outputs under models/runs/gen26_kboundary/
set -u
cd "$(dirname "$0")/.."
mkdir -p models/runs/gen26_kboundary
for S in 0 1 2; do
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 35-159 --N 3 --K 3 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --skip-vanilla --seed $S --threads 3 \
    --json-out models/runs/gen26_kboundary/k3_seed$S.json \
    --ckpt-dir models/runs/gen26_kboundary/k3_seed${S}_ckpts \
    > models/runs/gen26_kboundary/k3_seed$S.log 2>&1 &
done
wait
echo "GEN26_STEP1_DONE"
