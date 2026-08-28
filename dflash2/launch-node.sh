#!/usr/bin/env bash
set -euo pipefail

NODE_RANK="${1:?usage: launch-node.sh <0|1>}"
[[ "$NODE_RANK" == "0" || "$NODE_RANK" == "1" ]] || {
  echo "rank must be 0 or 1" >&2
  exit 2
}

RECIPE_DIR="${RECIPE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
IMAGE="${IMAGE:-ghcr.io/tonyd2wild/vllm-glm53-flash:sm121-v11-dflash2}"
TARGET_HOST_PATH="$(readlink -f "${TARGET_HOST_PATH:-$HOME/models/GLM-5.3-Flash-NVFP4}")"
DRAFT_HOST_PATH="$(readlink -f "${DRAFT_HOST_PATH:-$HOME/models/GLM-5.3-Flash-DFlash2}")"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-$HOME/models/vllm-cache-glm53-dflash2}"
API_KEY_FILE="${API_KEY_FILE:-$RECIPE_DIR/.vllm-api-key}"
HEAD_IP="${HEAD_IP:-10.100.120.2}"
WORKER_IP="${WORKER_IP:-10.100.120.1}"
FABRIC_IF="${FABRIC_IF:-enp1s0f1np1}"
FABRIC_HCA="${FABRIC_HCA:-rocep1s0f1}"
PORT="${PORT:-8889}"
MASTER_PORT="${MASTER_PORT:-29521}"
NAME="glm53-vllm-dflash2"

[[ -f "$TARGET_HOST_PATH/config.json" ]] || {
  echo "target checkpoint missing: $TARGET_HOST_PATH/config.json" >&2
  exit 1
}
[[ "$TARGET_HOST_PATH" == */hub/* ]] || {
  echo "target snapshot is not under a Hugging Face cache: $TARGET_HOST_PATH" >&2
  exit 1
}
TARGET_CACHE_ROOT="${TARGET_HOST_PATH%%/hub/*}"
TARGET_CACHE_REL="${TARGET_HOST_PATH#"$TARGET_CACHE_ROOT"/}"
TARGET_CONTAINER_PATH="/models/hf/$TARGET_CACHE_REL"
[[ -f "$DRAFT_HOST_PATH/config.json" ]] || {
  echo "draft checkpoint missing: $DRAFT_HOST_PATH/config.json" >&2
  exit 1
}
[[ -f "$RECIPE_DIR/inner-serve.sh" ]] || {
  echo "missing $RECIPE_DIR/inner-serve.sh" >&2
  exit 1
}
[[ -f "$RECIPE_DIR/redact-vllm-startup.py" ]] || {
  echo "missing $RECIPE_DIR/redact-vllm-startup.py" >&2
  exit 1
}
if [[ "$NODE_RANK" == "0" ]]; then
  [[ -s "$API_KEY_FILE" ]] || {
    echo "API key file missing: $API_KEY_FILE" >&2
    exit 1
  }
  HOST_IP="$HEAD_IP"
else
  HOST_IP="$WORKER_IP"
fi

mkdir -p "$CACHE_HOST_PATH" "$CACHE_HOST_PATH/vllm" "$CACHE_HOST_PATH/flashinfer"
docker rm -f "$NAME" >/dev/null 2>&1 || true

secret_mount=()
if [[ "$NODE_RANK" == "0" ]]; then
  secret_mount=(-v "$API_KEY_FILE:/run/secrets/vllm-api-key:ro")
fi

docker run --gpus all -d \
  --name "$NAME" --restart no \
  --network host --ipc host --shm-size 32g \
  --ulimit memlock=-1:-1 --cap-add IPC_LOCK \
  --device /dev/infiniband:/dev/infiniband \
  -v "$TARGET_CACHE_ROOT:/models/hf:ro" \
  -v "$DRAFT_HOST_PATH:/models/draft:ro" \
  -v "$CACHE_HOST_PATH:/cache" \
  -v "$CACHE_HOST_PATH/vllm:/root/.cache/vllm" \
  -v "$CACHE_HOST_PATH/flashinfer:/root/.cache/flashinfer" \
  -v "$RECIPE_DIR/inner-serve.sh:/opt/apollo/inner-serve.sh:ro" \
  -v "$RECIPE_DIR/redact-vllm-startup.py:/opt/apollo/redact-vllm-startup.py:ro" \
  "${secret_mount[@]}" \
  -e NODE_RANK="$NODE_RANK" \
  -e HEAD_IP="$HEAD_IP" \
  -e PORT="$PORT" \
  -e MASTER_PORT="$MASTER_PORT" \
  -e TARGET_CONTAINER_PATH="$TARGET_CONTAINER_PATH" \
  -e VLLM_HOST_IP="$HOST_IP" \
  -e HF_HOME=/cache/huggingface \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a \
  -e FLASHINFER_DISABLE_VERSION_CHECK=1 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_HCA="$FABRIC_HCA" -e NCCL_IB_GID_INDEX=3 \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ADDR_RANGE=10.100.120.0/24 \
  -e NCCL_SOCKET_IFNAME="$FABRIC_IF" \
  -e GLOO_SOCKET_IFNAME="$FABRIC_IF" \
  -e TP_SOCKET_IFNAME="$FABRIC_IF" \
  -e MN_IF_NAME="$FABRIC_IF" \
  -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 \
  -e NCCL_IB_MERGE_NICS=0 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 \
  --entrypoint /bin/bash \
  "$IMAGE" /opt/apollo/inner-serve.sh

sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep -F "$NAME"
