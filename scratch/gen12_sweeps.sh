#!/bin/bash
# gen12_sweeps orchestrator (pre-registered in experiments/gen12_sweeps.md).
# ARM_FLAGS = the gen11-selected arm's flags (recorded in the ledger's launch record).
# Grid: headline OD 62-97 (N3K1 x3 seeds; N3K2, N3K3, N2K1, N5K1 x seed 0) + held-out 35-159
# (same five cells x seed 0) + post-fix vanilla seeds 1,2 on 62-97 (item-3 completion).
set -u
cd "$(dirname "$0")/.."
OUT=models/runs/gen12_sweeps
mkdir -p "$OUT"
PY=.venv/bin/python
ARM_FLAGS=${ARM_FLAGS:-""}

run () {  # run <name> <od> <N> <K> <seed> [extra flags...]
  local name=$1 od=$2 n=$3 k=$4 seed=$5; shift 5
  PYTHONPATH=. $PY scripts/train_multiconvoy.py \
    --od "$od" --N "$n" --K "$k" --k-extra 8 --menu-select --band 0.15,0.95 \
    --attacker-mode smooth --fp-tau 0.05 --switch-every 200 --smooth-window 250 \
    --sorties 1200 --eval-every 100 --seed "$seed" --threads 3 \
    "$@" --json-out "$OUT/${name}.json" > "$OUT/${name}.log" 2>&1
}

stage () { echo "[gen12] $1  $(date)" >> "$OUT/orchestrator.log"; }

stage "stage 1: headline cell 62-97 N3K1 x 3 seeds (ARM_FLAGS='$ARM_FLAGS')"
for S in 0 1 2; do
  run "hl_N3K1_seed$S" 62-97 3 1 $S --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 \
      --ckpt-dir "$OUT/hl_N3K1_seed${S}_ckpts" $ARM_FLAGS &
done
wait
stage "stage 2: 62-97 K/N cells (N3K2, N3K3, N2K1)"
run "hl_N3K2" 62-97 3 2 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
run "hl_N3K3" 62-97 3 3 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
run "hl_N2K1" 62-97 2 1 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
wait
stage "stage 3: 62-97 N5K1 + vanilla seeds 1,2 (item-3 completion)"
run "hl_N5K1" 62-97 5 1 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
run "van_seed1" 62-97 3 1 1 --vanilla-only &
run "van_seed2" 62-97 3 1 2 --vanilla-only &
wait
stage "stage 4: held-out OD 35-159 cells (N3K1, N3K2, N2K1)"
run "ho_N3K1" 35-159 3 1 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
run "ho_N3K2" 35-159 3 2 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
run "ho_N2K1" 35-159 2 1 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
wait
stage "stage 5: held-out OD 35-159 cells (N3K3, N5K1)"
run "ho_N3K3" 35-159 3 3 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
run "ho_N5K1" 35-159 5 1 0 --fleet-route --leader-ent-frac 0.5 --leader-alpha-floor 0.20 $ARM_FLAGS &
wait
stage "ALL DONE"
touch "$OUT/DONE"
