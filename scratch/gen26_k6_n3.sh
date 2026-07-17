#!/bin/bash
# gen26 K=6 to n=3 (the pre-registered open gate): seeds 1,2 (seed 0 = 0.718 banked).
# WAITS for the gen27 control to finish first (no oversubscribe), full thread hygiene.
set -u
cd "$(dirname "$0")/.."
while pgrep -qf "no-window"; do sleep 60; done
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
for S in 1 2; do
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 71-33 --N 3 --K 6 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --greedy-br --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --skip-vanilla --seed $S --threads 3 \
    --json-out models/runs/gen26_kboundary/k6_seed$S.json \
    --ckpt-dir models/runs/gen26_kboundary/k6_seed${S}_ckpts \
    > models/runs/gen26_kboundary/k6_seed$S.log 2>&1 &
done
wait
echo "GEN26_K6_N3_DONE"
