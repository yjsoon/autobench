#!/usr/bin/env bash
# Serving benchmark for Gemma-4 ModelOpt FP4 checkpoints that require the Axion-style
# SGLang branch overlay (`gemma4-modelopt-ptq`) rather than a stock image.
#
# This follows the AxionML/Gemma-4-12B-NVFP4 model-card recipe: start from a newer SGLang
# nightly, swap to the Gemma-4 ModelOpt branch inside the container, pin transformers, then
# launch the OpenAI-compatible server and drive it with scripts/bench-serving.py.
#
# Usage:
#   bench-sglang-gemma4-modelopt-serving.sh <hf-model-path> [ctx] [conc] [num_prompts] [max_seconds] [max_tokens] [extra sglang args...]
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null || true; set +a

DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
IMAGE="${SGLANG_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed}"
SGLANG_BRANCH="${SGLANG_BRANCH:-gemma4-modelopt-ptq}"
TRANSFORMERS_REF="${TRANSFORMERS_REF:-1423d22f7a3b62e8c70ad67b58ec25cd9b675897}"
PORT=30000

MODEL="${1:?need an HF model path, e.g. AxionML/Gemma-4-12B-NVFP4}"
CTX="${2:-65536}"; CONC="${3:-32}"; NUMP="${4:-1000}"; MAXS="${5:-900}"; MAXTOK="${6:-256}"
shift 6 2>/dev/null || shift $#
EXTRA=("$@")
NAME="sgl-g4mo-$(echo "$MODEL" | tr -c 'A-Za-z0-9' '-')"
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> launch Gemma-4 ModelOpt SGLang overlay $MODEL (ctx=$CTX) extra=[${EXTRA[*]:-}]"
docker run -d --name "$NAME" --gpus all --ipc=host --shm-size 128g -p "$PORT:$PORT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/tiktoken_encodings:/tiktoken_encodings" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings" \
  "$IMAGE" bash -lc "
    set -euo pipefail
    cd /
    rm -rf /sgl-workspace/sglang
    git clone https://github.com/bzhng-development/sglang.git /sgl-workspace/sglang
    cd /sgl-workspace/sglang
    git checkout \"$SGLANG_BRANCH\"
    pip install \"git+https://github.com/huggingface/transformers.git@$TRANSFORMERS_REF\"
    python3 -m sglang.launch_server \
      --model-path \"$MODEL\" \
      --host 0.0.0.0 --port \"$PORT\" --context-length \"$CTX\" \
      ${EXTRA[*]}
  " >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> waiting for /health (Gemma-4 branch overlay can take several minutes) ..."
for i in $(seq 1 2400); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "    ready after ${i}s"; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! SGLang container exited during load:"; docker logs "$NAME" 2>&1 | tail -60; exit 1
  fi
  sleep 1
  [ "$i" = 2400 ] && { echo "!! health timeout"; docker logs "$NAME" 2>&1 | tail -60; exit 1; }
done

min_avail=$base_avail
( while :; do
    cur=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$cur" -lt "$min_avail" ] && min_avail=$cur
    echo "$min_avail" > /tmp/$NAME.minavail; sleep 10
  done ) &
sampler=$!

echo "==> run ShareGPT serving benchmark (conc=$CONC num=$NUMP max=${MAXS}s)"
python3 "$BENCH" --base-url "http://localhost:$PORT" --model "$MODEL" \
  --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" \
  --concurrency "$CONC" --max-tokens "$MAXTOK" || true

kill "$sampler" >/dev/null 2>&1 || true
min_avail=$(cat /tmp/$NAME.minavail 2>/dev/null || echo "$base_avail"); rm -f /tmp/$NAME.minavail
echo "MEM mem_gb=$(echo "$((base_avail - min_avail))" | gib_kb) mem_source=\"system MemAvailable delta (10s sampling)\""

if printf '%s\n' "${EXTRA[@]:-}" | grep -q "speculative"; then
  echo "==> SPEC-METRICS (acceptance):"
  docker logs "$NAME" 2>&1 | grep -iE "accept|spec" | tail -20 || true
fi
