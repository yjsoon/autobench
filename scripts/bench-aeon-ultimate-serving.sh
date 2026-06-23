#!/usr/bin/env bash
# Serving benchmark for the AEON custom "vllm-ultimate" DFlash recipe.
#
# This intentionally runs the untrusted third-party image with NO credentials passed through and
# mounts both the model snapshot and the external drafter read-only. It exposes the same OpenAI-
# compatible API that scripts/bench-serving.py expects.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
IMAGE="${AEON_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"
PORT=8000
MODEL_NAME="${AEON_SERVED_MODEL_NAME:-aeon-ultimate}"
MODEL_REPO_ROOT="${AEON_MODEL_REPO_ROOT:-$HOME/.cache/huggingface/hub/models--AEON-7--Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS}"
MODEL_SNAPSHOT_PATH="${AEON_MODEL_SNAPSHOT_PATH:-/hfroot/snapshots/4ea4a8d3b8beee13b4e883748bab6221f119cbb0}"
DRAFTER_DIR="${AEON_DRAFTER_DIR:-$HOME/models/qwen36-27b-dflash}"

CTX="${1:-65536}"; CONC="${2:-32}"; NUMP="${3:-1000}"; MAXS="${4:-900}"; MAXTOK="${5:-256}"
shift 5 2>/dev/null || shift $#
EXTRA=("$@")
NAME="aeon-ultimate-c${CONC}"
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> launch AEON ultimate DFlash (ctx=$CTX conc=$CONC) extra=[${EXTRA[*]:-}]"
docker run -d --name "$NAME" --gpus all --ipc=host -p "$PORT:$PORT" \
  -v "$MODEL_REPO_ROOT:/hfroot:ro" \
  -v "$DRAFTER_DIR:/drafter:ro" \
  --entrypoint vllm "$IMAGE" \
  serve "$MODEL_SNAPSHOT_PATH" --served-model-name "$MODEL_NAME" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$CTX" \
  --quantization modelopt --mamba-cache-dtype float16 --mamba-block-size 256 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --limit-mm-per-prompt '{"image":4,"video":2}' --mm-encoder-tp-mode data \
  --gpu-memory-utilization 0.85 --max-num-seqs "$CONC" --max-num-batched-tokens 16384 \
  --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
  --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":12}' \
  "${EXTRA[@]}" >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> waiting for /health (custom container can take several minutes) ..."
for i in $(seq 1 1800); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "    ready after ${i}s"; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! AEON container exited during load:"; docker logs "$NAME" 2>&1 | tail -60; exit 1
  fi
  sleep 1
  [ "$i" = 1800 ] && { echo "!! health timeout"; docker logs "$NAME" 2>&1 | tail -60; exit 1; }
done

min_avail=$base_avail
( while :; do
    cur=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$cur" -lt "$min_avail" ] && min_avail=$cur
    echo "$min_avail" > /tmp/$NAME.minavail; sleep 10
  done ) &
sampler=$!

echo "==> run ShareGPT serving benchmark (conc=$CONC num=$NUMP max=${MAXS}s)"
python3 "$BENCH" --base-url "http://localhost:$PORT" --model "$MODEL_NAME" \
  --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" \
  --concurrency "$CONC" --max-tokens "$MAXTOK" || true

kill "$sampler" >/dev/null 2>&1 || true
min_avail=$(cat /tmp/$NAME.minavail 2>/dev/null || echo "$base_avail"); rm -f /tmp/$NAME.minavail
echo "MEM mem_gb=$(echo "$((base_avail - min_avail))" | gib_kb) mem_source=\"system MemAvailable delta (10s sampling)\""

echo "==> SPEC-METRICS (acceptance):"
docker logs "$NAME" 2>&1 | grep -iE "accept|spec.?decode|num_spec|draft|dflash" | tail -20 || true
