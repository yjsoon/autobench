#!/usr/bin/env bash
# Serving benchmark for one HF model via SGLang, driven by scripts/bench-serving.py
# against the ShareGPT workload.
#
# Strix Halo port: default image is the locally-built strix-halo-sglang:dev
# (https://github.com/JeremiahM37/strix-halo-sglang — build it first). ROCm devices
# are passed through (/dev/kfd + /dev/dri) instead of --gpus all. On gfx1151 you
# generally need: --mem-fraction-static 0.5 --attention-backend triton --disable-cuda-graph
# (pass as extra args). The tunableop cache mount matters: without it single-stream
# throughput roughly halves.
#
# SGLang exposes an OpenAI-compatible API on :30000, so the same client works as for
# llama.cpp / vLLM. Headline memory = system MemAvailable delta from idle (10s sampling).
#
# Usage: bench-sglang-serving.sh <hf-model-path> [ctx] [conc] [num_prompts] [max_seconds] [max_tokens] [extra sglang args...]
#   e.g. bench-sglang-serving.sh Qwen/Qwen3.5-4B 8192 8 1000 900 256 \
#          --mem-fraction-static 0.5 --attention-backend triton --disable-cuda-graph
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null || true; set +a

DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
# Default to the locally-built Strix Halo image; override with SGLANG_IMAGE.
IMAGE="${SGLANG_IMAGE:-strix-halo-sglang:dev}"
PORT=30000

MODEL="${1:?need an HF model path, e.g. openai/gpt-oss-20b}"
CTX="${2:-65536}"; CONC="${3:-32}"; NUMP="${4:-1000}"; MAXS="${5:-900}"; MAXTOK="${6:-256}"
shift 6 2>/dev/null || shift $#
EXTRA=("$@")   # any extra sglang.launch_server flags
NAME="sgl-$(echo "$MODEL" | tr -c 'A-Za-z0-9' '-')"
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> launch SGLang $MODEL (ctx=$CTX) extra=[${EXTRA[*]:-}]"
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri \
  --ipc=host --shm-size 32g --security-opt seccomp=unconfined -p "$PORT:$PORT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/strix-halo-sglang-tunableop:/root/.tunableop" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "SGLANG_FORCE_NATIVE_LAYERNORM=1" \
  "$IMAGE" python3 -m sglang.launch_server --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --context-length "$CTX" "${EXTRA[@]}" >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> waiting for /health (model download+load can be minutes) ..."
for i in $(seq 1 1200); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "    ready after ${i}s"; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! SGLang container exited during load:"; docker logs "$NAME" 2>&1 | tail -40; exit 1
  fi
  sleep 1
  [ "$i" = 1200 ] && { echo "!! health timeout"; docker logs "$NAME" 2>&1 | tail -40; exit 1; }
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

# Speculative-decoding acceptance metrics (EAGLE3/NEXTN runs) — from the server log before teardown.
# SGLang logs e.g. "Accept length: X.XX" / "accept_length" per decode.
if printf '%s\n' "${EXTRA[@]:-}" | grep -q "speculative"; then
  echo "==> SPEC-METRICS (acceptance):"
  docker logs "$NAME" 2>&1 | grep -iE "accept|spec" | tail -20 || true
fi
