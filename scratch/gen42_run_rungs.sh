#!/bin/zsh
# gen42 rung runner (2026-08-07): mounts each Qwen3.5 rung on the w05 box (our account,
# draft flags from /home/killian/ladder_prep/models_draft.json), waits for readiness, runs
# the battery from this Mac direct-to-port (the gateway loads its registry at startup only),
# stops the rung, then proceeds. Final step restores llama-3.3-70b from its captured
# command line (/home/killian/ladder_prep/llama_restore_cmdline.txt). Requires llama to be
# DOWN before the first mount (VRAM). Pre-registered act: experiments/gen42_capability_ladder.md.
set -u
cd "$(dirname "$0")/.."
BOX=killian@100.88.32.88
HOSTN=cv-iits-w05.tail5b8d80.ts.net
PY=../sacred/.venv/bin/python
VLLM=/home/llm/vllm-env/bin/vllm
ENVF=/home/llm/vllm-server/vllm.env
PREP=/home/killian/ladder_prep

RUNGS=(
  "qwen35-2b|cyankiwi/Qwen3.5-2B-AWQ-4bit|8005|0.10"
  "qwen35-4b|cyankiwi/Qwen3.5-4B-AWQ-4bit|8006|0.15"
  "qwen35-9b|cyankiwi/Qwen3.5-9B-AWQ-4bit|8007|0.20"
  "qwen35-27b|cyankiwi/Qwen3.5-27B-AWQ-4bit|8008|0.37"
)

wait_ready() {  # $1 port; up to 20 min
  for i in $(seq 1 120); do
    code=$(curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer iits-local-key" \
      "http://$HOSTN:$1/v1/models" 2>/dev/null || echo 000)
    [ "$code" = "200" ] && return 0
    sleep 10
  done
  return 1
}

stop_rung() {  # $1 name
  ssh $BOX "PID=\$(cat $PREP/serve_$1.pid 2>/dev/null); [ -n \"\$PID\" ] && PGID=\$(ps -o pgid= -p \$PID | tr -d ' ') && kill -TERM -\"\$PGID\" 2>/dev/null; sleep 15; [ -n \"\$PID\" ] && ps -p \$PID >/dev/null 2>&1 && echo '$1 STILL ALIVE' || echo '$1 stopped'"
}

for spec in $RUNGS; do
  name=${spec%%|*}; rest=${spec#*|}; repo=${rest%%|*}; rest=${rest#*|}; port=${rest%%|*}; util=${rest##*|}
  echo "=== RUNG $name ($repo) port $port util $util  $(date) ==="
  ssh $BOX "source $ENVF && setsid nohup $VLLM serve $repo --tensor-parallel-size 2 --host 0.0.0.0 --port $port --api-key iits-local-key --served-model-name $name --max-model-len 65536 --gpu-memory-utilization $util --enforce-eager --max-num-seqs 16 --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 > $PREP/serve_$name.log 2>&1 & echo \$! > $PREP/serve_$name.pid; cat $PREP/serve_$name.pid"
  if wait_ready $port; then
    echo "--- $name READY, battery starts $(date) ---"
    OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 PYTHONPATH=. $PY scratch/gen42_battery.py \
      --rung $name --base "http://$HOSTN:$port/v1" || echo "--- $name battery reported failures ---"
  else
    echo "--- $name NEVER BECAME READY (20 min); serve log tail: ---"
    ssh $BOX "tail -5 $PREP/serve_$name.log"
  fi
  stop_rung $name
done

echo "=== RESTORING residents (stack: gateway + llama; qwen from captured cmdline)  $(date) ==="
ssh $BOX "cd /home/llm/vllm-server && ./start.sh start > $PREP/stack_restart.log 2>&1; sleep 5; tail -3 $PREP/stack_restart.log"
ssh $BOX "source $ENVF && setsid nohup \$(cat $PREP/qwen_restore_cmdline.txt) > $PREP/serve_qwen_restored.log 2>&1 & echo \$! > $PREP/serve_qwen_restored.pid"
if wait_ready 8002; then echo "llama RESTORED and serving"; else echo "llama RESTORE FAILED - manual attention needed"; fi
if wait_ready 8001; then echo "qwen RESTORED and serving"; else echo "qwen RESTORE FAILED - manual attention needed"; fi
ssh $BOX "/home/llm/vllm-env/bin/python -c \"
import torch
for i in range(2):
    f,t=torch.cuda.mem_get_info(i); print(f'GPU{i} free {f/2**30:.1f} GiB of {t/2**30:.1f}')
\" 2>/dev/null; curl -s http://localhost:8080/health -w 'gateway %{http_code}\n' -o /dev/null"
echo "=== RUNG RUNNER DONE $(date) ==="
