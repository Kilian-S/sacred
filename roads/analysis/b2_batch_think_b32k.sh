#!/bin/bash
# B2 LLM benchmark batch in thinking mode, register (b) at a uniform 32k token cap.
# One worker per cell, posting direct to the vLLM port. Idempotent: existing outputs
# are skipped.
set -u; cd "$(dirname "$0")/.."
# The recorded run posted direct to the vLLM port (8001) rather than the gateway (8080);
# point SACRED_LLM_BASE at that port to reproduce it.
BASE="${SACRED_LLM_BASE:?set SACRED_LLM_BASE to the OpenAI-compatible endpoint}"; KEY="${SACRED_LLM_KEY:?set SACRED_LLM_KEY}"; MODEL="qwen3-27b"
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
