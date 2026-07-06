#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-8080}"
HOST="${HOST:-0.0.0.0}"
PARALLEL="${PARALLEL:-1}"
CONTEXT="${CONTEXT:-65536}"
N_GPU_LAYERS="${N_GPU_LAYERS:-99}"
MODEL_ROOT="${MODEL_ROOT:-$HOME/.lmstudio/models}"
MODEL_REL="${MODEL_REL:-lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf}"
MODEL_ALIAS="${MODEL_ALIAS:-qwen3.6-35b-a3b-q4km}"
IMAGE="${IMAGE:-ghcr.io/ggml-org/llama.cpp:full-vulkan}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen-local-code}"
DETACH="${DETACH:-0}"

MODEL_PATH="$MODEL_ROOT/$MODEL_REL"

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required to run the llama.cpp Vulkan server" >&2
  exit 1
fi

if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "Container '$CONTAINER_NAME' is already running." >&2
  echo "Use: docker logs -f $CONTAINER_NAME" >&2
  exit 0
fi

docker_args=(
  run --rm --name "$CONTAINER_NAME"
  --device /dev/dri --device /dev/kfd
  -p "$PORT:8080"
  -v "$MODEL_ROOT":/models:ro
)

if [[ "$DETACH" == "1" || "$DETACH" == "true" ]]; then
  docker_args+=(-d)
fi

docker_args+=(
  "$IMAGE"
  --server
  -m "/models/$MODEL_REL"
  --alias "$MODEL_ALIAS"
  -ngl "$N_GPU_LAYERS"
  -c "$CONTEXT"
  --parallel "$PARALLEL"
  -cb
  --reasoning off
  --reasoning-format deepseek
  --host "$HOST"
  --port 8080
)

exec docker "${docker_args[@]}"
