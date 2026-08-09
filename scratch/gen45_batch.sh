#!/bin/bash
# gen45 ceremony launcher (ledger experiments/gen45_unified_corridor.md; Kilian launches).
#   bash scratch/gen45_batch.sh attempt   -> 3 runs, seeds 0/1/2, dev-test diagnostics
#   bash scratch/gen45_batch.sh confirm   -> 4 runs: fresh seeds 10/11/12 + BLINDED control
#                                            (seed 10), all --eval-gated; the citable wave
# Children are spawned by this bash script in the foreground shell, so the interactive-zsh
# background nice(5) trap never applies; verify with: ps -o pid,nice,command | grep gen45
set -e
cd "$(dirname "$0")/.."
PY=../sacred/.venv/bin/python
export OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE PYTHONPATH=.
OUT=models/runs/gen45_unified
mkdir -p "$OUT"
case "$1" in
attempt)
  for S in 0 1 2; do
    $PY scripts/train_gen45_unified.py --sorties 16000 --seed $S --threads 1 \
      --json-out $OUT/attempt_seed$S.json --ckpt-dir $OUT/attempt_seed${S}_ckpts \
      > $OUT/attempt_seed$S.log 2>&1 &
  done
  wait ;;
confirm)
  for S in 10 11 12; do
    $PY scripts/train_gen45_unified.py --sorties 16000 --seed $S --threads 1 --eval-gated \
      --json-out $OUT/confirm_seed$S.json --ckpt-dir $OUT/confirm_seed${S}_ckpts \
      > $OUT/confirm_seed$S.log 2>&1 &
  done
  $PY scripts/train_gen45_unified.py --sorties 16000 --seed 10 --threads 1 --eval-gated --blind \
    --json-out $OUT/blind_seed10.json --ckpt-dir $OUT/blind_seed10_ckpts \
    > $OUT/blind_seed10.log 2>&1 &
  wait ;;
*)
  echo "usage: bash scratch/gen45_batch.sh attempt|confirm"; exit 1 ;;
esac
echo "[gen45] wave '$1' complete"
