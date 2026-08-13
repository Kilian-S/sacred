#!/bin/bash
# B2 THINKING-MODE rerun (pre-registered 2026-08-13): qwen3-27b only, all three cells as
# three concurrent streams. Protocol identical to the banked cells except --thinking and
# max-tokens 16000 (forced co-change, the 5c precedent). Idempotent: existing outputs skip.
set -u; cd "$(dirname "$0")/.."
BASE="http://cv-iits-w05.tail5b8d80.ts.net:8080/v1"; KEY="iits-local-key"; MODEL="qwen3-27b"
run_cell () {
  local OD=$1 CITY=$2 TAG=$3
  local DIR="models/runs/b2_llm/batch_${TAG}_think"
  mkdir -p "$DIR"
  for REG in a b c; do
    local N=10; [ "$REG" = "c" ] && N=5
    for S in $(seq 0 $((N-1))); do
      local OUT="${DIR}/${MODEL}_${REG}_seed${S}.json"
      [ -s "$OUT" ] && continue
      for ATT in 1 2; do
        PYTHONPATH=. .venv/bin/python scratch/b2_llm_benchmark.py \
          --od "$OD" --city "$CITY" --K 1 --provider openai --base "$BASE" --key "$KEY" \
          --model "$MODEL" --thinking --register "$REG" --seed "$S" --temperature 0.7 \
          --max-tokens 16000 --json-out "$OUT" >> "${DIR}/${MODEL}.log" 2>&1 && break
        echo "RETRY ${TAG} ${REG} s${S} att${ATT}" >> "${DIR}/${MODEL}.log"; sleep 20
      done
    done
  done
  echo "CELL_${TAG}_DONE"
}
run_cell 71-33 kaliningrad 7133 &
run_cell 35-159 kaliningrad 35159 &
run_cell 249-95 gdansk gdansk &
wait
echo "B2_THINK_DONE"
