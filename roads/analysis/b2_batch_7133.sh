#!/bin/bash
# B2 LLM benchmark batch for the 71-33 cell at K=1.
set -u; cd "$(dirname "$0")/.."
mkdir -p models/runs/b2_llm/batch_7133
BASE="${SACRED_LLM_BASE:?set SACRED_LLM_BASE to the OpenAI-compatible endpoint}"; KEY="${SACRED_LLM_KEY:?set SACRED_LLM_KEY}"; OD="71-33"
run_model () {
  local MODEL=$1
  for REG in a b c; do
    local N=10; [ "$REG" = "c" ] && N=5
    for S in $(seq 0 $((N-1))); do
      local OUT="models/runs/b2_llm/batch_7133/${MODEL}_${REG}_seed${S}.json"
      [ -s "$OUT" ] && continue
      for ATT in 1 2; do
        PYTHONPATH=. .venv/bin/python analysis/b2_llm_benchmark.py \
          --od "$OD" --city kaliningrad --K 1 --provider openai --base "$BASE" --key "$KEY" --model "$MODEL" \
          --register "$REG" --seed "$S" --temperature 0.7 --max-tokens 12000 --json-out "$OUT" \
          >> "models/runs/b2_llm/batch_7133/${MODEL}.log" 2>&1 && break
        echo "RETRY ${MODEL} ${REG} s${S} att${ATT}" >> "models/runs/b2_llm/batch_7133/${MODEL}.log"; sleep 20
      done
    done
  done
}
run_model llama-3.3-70b &
run_model qwen3-27b &
wait
echo "B2_7133_DONE"
