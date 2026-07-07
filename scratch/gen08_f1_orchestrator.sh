#!/usr/bin/env bash
# gen08 F1 (waves A + C): 12 interdiction cells, attacker=LATEST, 3 concurrent.
# Pre-registration + method rationale: experiments/gen08_interdiction.md (F1 launch record).
# Launch detached:  ( nohup bash scratch/gen08_f1_orchestrator.sh > \
#                     models/runs/gen08_interdiction_I3/f1_orchestrator.log 2>&1 & )
set -u
cd /Users/kilian/Kilian/ICL/Thesis/code/sacred || exit 1
OUT=models/runs/gen08_interdiction_I3
mkdir -p "$OUT"
MAXJOBS=3

run_cell() {
  local tag="$1"; shift
  PYTHONPATH=. .venv/bin/python scripts/train_interdiction.py "$@" \
    --sorties 3000 --eval-every 250 --attacker-mode latest \
    --json-out "$OUT/${tag}.json" > "$OUT/${tag}.log" 2>&1
}

declare -a CELLS
for k in 1 2 3; do for s in 0 1 2; do CELLS+=("A_K${k}_seed${s}|--od 33-71 --K ${k} --seed ${s}"); done; done
for s in 0 1 2; do CELLS+=("C_seed${s}|--od 110-135 --K 1 --seed ${s} --edge-vuln-band 0.15,0.95"); done

echo "F1 launch: ${#CELLS[@]} cells, max ${MAXJOBS} concurrent, start $(date '+%Y-%m-%d %H:%M:%S')"
for cell in "${CELLS[@]}"; do
  tag="${cell%%|*}"; args="${cell##*|}"
  while [ "$(jobs -rp | wc -l | tr -d ' ')" -ge "$MAXJOBS" ]; do sleep 5; done
  echo "  [$(date '+%H:%M:%S')] launch ${tag} :: ${args}"
  # shellcheck disable=SC2086
  run_cell "$tag" $args &
done
wait
echo "F1 DONE: all ${#CELLS[@]} cells complete, end $(date '+%Y-%m-%d %H:%M:%S')"
