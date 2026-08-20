#!/bin/bash
# B2 overnight batch: the 35-159 cell, llama + qwen concurrent (one stream each).
# Ledger: experiments/b2_llm_benchmark.md launch record 2026-07-16.
set -u
cd "$(dirname "$0")/.."
mkdir -p models/runs/b2_llm/batch_35159
BASE="http://100.88.32.88:8080/v1"; KEY="iits-local-key"
run_model () {
  local MODEL=$1
  for REG in a b c; do
    local N=10; [ "$REG" = "c" ] && N=5
    for S in $(seq 0 $((N-1))); do
      local OUT="models/runs/b2_llm/batch_35159/${MODEL}_${REG}_seed${S}.json"
      [ -s "$OUT" ] && continue
      for ATTEMPT in 1 2; do
        PYTHONPATH=. .venv/bin/python analysis/b2_llm_benchmark.py \
          --provider openai --base "$BASE" --key "$KEY" --model "$MODEL" \
          --register "$REG" --seed "$S" --temperature 0.7 --max-tokens 12000 \
          --json-out "$OUT" \
          >> "models/runs/b2_llm/batch_35159/${MODEL}.log" 2>&1 && break
        echo "RETRY ${MODEL} ${REG} seed${S} (attempt ${ATTEMPT} failed)" >> "models/runs/b2_llm/batch_35159/${MODEL}.log"
        sleep 20
      done
    done
  done
}
run_model llama-3.3-70b &
run_model qwen3-27b &
wait
for T in 0.3 0.8; do
  for S in 0 1 2 3 4; do
    OUT="models/runs/b2_llm/batch_35159/llama_b_t${T}_seed${S}.json"
    [ -s "$OUT" ] && continue
    PYTHONPATH=. .venv/bin/python analysis/b2_llm_benchmark.py \
      --provider openai --base "$BASE" --key "$KEY" --model llama-3.3-70b \
      --register b --seed "$S" --temperature "$T" --max-tokens 12000 \
      --json-out "$OUT" >> models/runs/b2_llm/batch_35159/llama-3.3-70b.log 2>&1
  done
done
echo "B2_BATCH_35159_DONE"
