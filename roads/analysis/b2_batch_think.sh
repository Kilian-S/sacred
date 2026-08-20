#!/bin/bash
# B2 THINKING-MODE rerun (pre-registered 2026-08-13): qwen3-27b only, all three cells.
# v2 (same day, BEFORE any result existed): the first launch ran one stream per cell and
# projected ~25 h; conversations are independent across registers and episodes, so this
# version runs 5 workers per cell (a | b | c-seeds 0,1 | c-seeds 2,3 | c-seed 4) = 15
# concurrent streams, an ops change only (request bodies unchanged). Idempotent: existing
# outputs skip. Per-call timeout 1800 s in thinking mode (harness).
set -u; cd "$(dirname "$0")/.."
BASE="http://cv-iits-w05.tail5b8d80.ts.net:8080/v1"; KEY="iits-local-key"; MODEL="qwen3-27b"
worker () {
  local OD=$1 CITY=$2 TAG=$3 REG=$4; shift 4
  local DIR="models/runs/b2_llm/batch_${TAG}_think"
  mkdir -p "$DIR"
  for S in "$@"; do
    local OUT="${DIR}/${MODEL}_${REG}_seed${S}.json"
    [ -s "$OUT" ] && continue
    for ATT in 1 2; do
      PYTHONPATH=. .venv/bin/python analysis/b2_llm_benchmark.py \
        --od "$OD" --city "$CITY" --K 1 --provider openai --base "$BASE" --key "$KEY" \
        --model "$MODEL" --thinking --register "$REG" --seed "$S" --temperature 0.7 \
        --max-tokens 16000 --json-out "$OUT" >> "${DIR}/${MODEL}.log" 2>&1 && break
      echo "RETRY ${TAG} ${REG} s${S} att${ATT}" >> "${DIR}/${MODEL}.log"; sleep 20
    done
  done
}
run_cell () {
  local OD=$1 CITY=$2 TAG=$3
  worker "$OD" "$CITY" "$TAG" a 0 1 2 3 4 5 6 7 8 9 &
  sleep 2
  worker "$OD" "$CITY" "$TAG" b 0 1 2 3 4 5 6 7 8 9 &
  sleep 2
  worker "$OD" "$CITY" "$TAG" c 0 1 &
  sleep 2
  worker "$OD" "$CITY" "$TAG" c 2 3 &
  sleep 2
  worker "$OD" "$CITY" "$TAG" c 4 &
  wait
  echo "CELL_${TAG}_DONE"
}
run_cell 71-33 kaliningrad 7133 &
sleep 3
run_cell 35-159 kaliningrad 35159 &
sleep 3
run_cell 249-95 gdansk gdansk &
wait
echo "B2_THINK_DONE"
