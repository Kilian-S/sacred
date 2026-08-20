#!/bin/bash
# B2 thinking rerun, register (b) at the amended uniform 32k cap (2026-08-13; the 5c-class
# co-change, ledger amendment). v4: DIRECT-TO-PORT (qwen vllm on :8001, Kilian's authorisation;
# bypasses the gateway proxy window that 502'd long traces). Three workers, one per cell.
# Idempotent: existing outputs skip.
set -u; cd "$(dirname "$0")/.."
BASE="http://cv-iits-w05.tail5b8d80.ts.net:8001/v1"; KEY="iits-local-key"; MODEL="qwen3-27b"
worker () {
  local OD=$1 CITY=$2 TAG=$3; shift 3
  local DIR="models/runs/b2_llm/batch_${TAG}_think"
  mkdir -p "$DIR"
  for S in "$@"; do
    local OUT="${DIR}/${MODEL}_b_seed${S}.json"
    [ -s "$OUT" ] && continue
    for ATT in 1 2; do
      PYTHONPATH=. .venv/bin/python analysis/b2_llm_benchmark.py \
        --od "$OD" --city "$CITY" --K 1 --provider openai --base "$BASE" --key "$KEY" \
        --model "$MODEL" --thinking --register b --seed "$S" --temperature 0.7 \
        --max-tokens 32000 --json-out "$OUT" >> "${DIR}/${MODEL}_b32k.log" 2>&1 && break
      echo "RETRY ${TAG} b s${S} att${ATT}" >> "${DIR}/${MODEL}_b32k.log"; sleep 20
    done
  done
}
worker 71-33 kaliningrad 7133 0 1 2 3 4 5 6 7 8 9 &
worker 35-159 kaliningrad 35159 0 1 2 3 4 5 6 7 8 9 &
worker 249-95 gdansk gdansk 0 1 2 3 4 5 6 7 8 9 &
wait
echo "B2_THINK_B32K_DONE"
