#!/usr/bin/env bash
set -euo pipefail

RECIPE_DIR="${RECIPE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
WORKER_SSH="${WORKER_SSH:-apollo-2}"
PORT="${PORT:-8889}"
API_KEY_FILE="${API_KEY_FILE:-$RECIPE_DIR/.vllm-api-key}"

case "${1:-start}" in
  stop)
    docker stop glm53-vllm-dflash2 >/dev/null 2>&1 || true
    ssh "$WORKER_SSH" 'docker stop glm53-vllm-dflash2 >/dev/null 2>&1 || true'
    ;;
  status)
    docker inspect -f 'head {{.State.Status}} restart={{.RestartCount}}' glm53-vllm-dflash2 2>/dev/null || true
    ssh "$WORKER_SSH" "docker inspect -f 'worker {{.State.Status}} restart={{.RestartCount}}' glm53-vllm-dflash2 2>/dev/null || true"
    key="$(< "$API_KEY_FILE")"
    curl -sS -o /dev/null -w 'health=%{http_code}\n' -H "Authorization: Bearer $key" "http://127.0.0.1:$PORT/health" || true
    ;;
  start)
    sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true
    ssh "$WORKER_SSH" "sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || true"
    ssh "$WORKER_SSH" "$RECIPE_DIR/launch-node.sh 1"
    sleep 20
    "$RECIPE_DIR/launch-node.sh" 0
    ;;
  *)
    echo "usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac
