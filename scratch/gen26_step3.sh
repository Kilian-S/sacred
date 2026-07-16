#!/bin/bash
# gen26 step 3: K=5 (3 seeds, headline) then K=6 (1 seed, saturation boundary) on 71-33,
# greedy-BR mode (ledger: experiments/gen26_kboundary.md, step-3 amendment).
set -u
cd "$(dirname "$0")/.."
mkdir -p models/runs/gen26_kboundary
for S in 0 1 2; do
  PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
    --od 71-33 --N 3 --K 5 --k-extra 8 --menu-select --band 0.15,0.95 \
    --fleet-route --attacker-mode smooth --greedy-br --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
    --skip-vanilla --seed $S --threads 3 \
    --json-out models/runs/gen26_kboundary/k5_seed$S.json \
    --ckpt-dir models/runs/gen26_kboundary/k5_seed${S}_ckpts \
    > models/runs/gen26_kboundary/k5_seed$S.log 2>&1 &
done
wait
PYTHONPATH=. .venv/bin/python scripts/train_multiconvoy.py \
  --od 71-33 --N 3 --K 6 --k-extra 8 --menu-select --band 0.15,0.95 \
  --fleet-route --attacker-mode smooth --greedy-br --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
  --leader-ent-frac 0.5 --leader-alpha-floor 0.20 --sorties 1200 --eval-every 100 \
  --skip-vanilla --seed 0 --threads 4 \
  --json-out models/runs/gen26_kboundary/k6_seed0.json \
  --ckpt-dir models/runs/gen26_kboundary/k6_seed0_ckpts \
  > models/runs/gen26_kboundary/k6_seed0.log 2>&1
echo "GEN26_STEP3_DONE"
