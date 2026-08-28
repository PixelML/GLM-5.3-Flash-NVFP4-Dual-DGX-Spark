#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${NODE_RANK:?NODE_RANK is required}"
HEAD_IP="${HEAD_IP:?HEAD_IP is required}"
PORT="${PORT:-8889}"
MASTER_PORT="${MASTER_PORT:-29521}"
TARGET_CONTAINER_PATH="${TARGET_CONTAINER_PATH:?TARGET_CONTAINER_PATH is required}"

python3 /opt/apollo/redact-vllm-startup.py

args=(
  serve "$TARGET_CONTAINER_PATH"
  --served-model-name apollo-glm-5.3-flash glm-5.3-flash
  --host 0.0.0.0
  --port "$PORT"
  --trust-remote-code
  --tensor-parallel-size 2
  --distributed-executor-backend mp
  --nnodes 2
  --node-rank "$NODE_RANK"
  --master-addr "$HEAD_IP"
  --master-port "$MASTER_PORT"
  --gpu-memory-utilization 0.85
  --max-model-len 262144
  --max-num-seqs 6
  --block-size 2304
  --moe-backend marlin
  --kv-cache-dtype fp8_e4m3
  --kv-cache-memory 3221225472
  --speculative-config '{"method":"dflash","model":"/models/draft","num_speculative_tokens":7}'
  --enforce-eager
  --tool-call-parser glm47
  --enable-auto-tool-choice
  --reasoning-parser glm45
  --default-chat-template-kwargs '{"enable_thinking":false}'
  --skip-mm-profiling
)

if [[ "$NODE_RANK" == "0" ]]; then
  [[ -s /run/secrets/vllm-api-key ]] || {
    echo "missing /run/secrets/vllm-api-key" >&2
    exit 1
  }
  export VLLM_API_KEY
  VLLM_API_KEY="$(< /run/secrets/vllm-api-key)"
else
  args+=(--headless)
fi

exec vllm "${args[@]}"
