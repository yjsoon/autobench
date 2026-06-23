#!/usr/bin/env bash
# Serving benchmark for one GGUF: launch llama-server (NGC dispatcher image),
# drive it with the ShareGPT workload via scripts/bench-serving.py, sample memory.
#
# Headline memory = system MemAvailable delta from an idle baseline (10s sampling) —
# nvidia-smi gives no memory on the GB10 (unified) and docker stats undercounts CUDA
# unified allocations.
#
# Usage: bench-llamacpp-serving.sh <model.gguf under /home/gauravmm/models> \
#          [ctx] [concurrency] [num_prompts] [max_seconds] [max_tokens] [ngl] [extra llama-server args...]
#   Extra args pass straight to llama-server — e.g. speculative decoding (Gemma 4 MTP drafter):
#     ... 99 --model-draft /models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on
set -euo pipefail

MODELS_DIR=/home/gauravmm/models
DATASET="$(cd "$(dirname "$0")/.." && pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(cd "$(dirname "$0")" && pwd)/bench-serving.py"
IMAGE=ghcr.io/ggml-org/llama.cpp:full-cuda
PORT=8081

FILE="${1:?need a gguf filename under $MODELS_DIR}"
CTX="${2:-8192}"; CONC="${3:-32}"; NUMP="${4:-1000}"; MAXS="${5:-900}"; MAXTOK="${6:-256}"; NGL="${7:-99}"
shift 7 2>/dev/null || shift $#
EXTRA=("$@")   # extra llama-server flags (e.g. --model-draft … --spec-type draft-mtp …)
NAME="serve-$(echo "$FILE" | tr -c 'A-Za-z0-9' '-')"
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)   # kB, idle baseline
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> launch llama-server $FILE (ctx=$CTX ngl=$NGL) extra=[${EXTRA[*]:-}]"
docker run -d --name "$NAME" --gpus all -p "$PORT:$PORT" \
  -v "$MODELS_DIR":/models:ro "$IMAGE" \
  --server -m "/models/$FILE" -ngl "$NGL" -c "$CTX" \
  --parallel "$CONC" -cb --host 0.0.0.0 --port "$PORT" "${EXTRA[@]}" >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

# Wait for health (model load can take a while for big files — multi-100GB GGUFs need >180s,
# so the timeout is env-configurable via HEALTH_TIMEOUT).
HT="${HEALTH_TIMEOUT:-180}"
echo "==> waiting for /health (timeout ${HT}s) ..."
for i in $(seq 1 "$HT"); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "    ready after ${i}s"; break
  fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! server container exited during load:"; docker logs "$NAME" 2>&1 | tail -30; exit 1
  fi
  sleep 1
  [ "$i" = "$HT" ] && { echo "!! health timeout"; docker logs "$NAME" 2>&1 | tail -30; exit 1; }
done

# Sample MemAvailable in the background during the benchmark.
min_avail=$base_avail
sample_stop=0
( while [ "$sample_stop" = 0 ]; do
    cur=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$cur" -lt "$min_avail" ] && min_avail=$cur
    echo "$min_avail" > /tmp/$NAME.minavail
    sleep 10
  done ) &
sampler=$!

echo "==> run ShareGPT serving benchmark (conc=$CONC num=$NUMP max=${MAXS}s)"
python3 "$BENCH" --base-url "http://localhost:$PORT" --model "$FILE" \
  --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" \
  --concurrency "$CONC" --max-tokens "$MAXTOK" || true

kill "$sampler" >/dev/null 2>&1 || true
min_avail=$(cat /tmp/$NAME.minavail 2>/dev/null || echo "$base_avail")
rm -f /tmp/$NAME.minavail
sys_delta_gib=$(echo "$((base_avail - min_avail))" | gib_kb)
echo "MEM mem_gb=$sys_delta_gib mem_source=\"system MemAvailable delta (10s sampling)\""

# Speculative-decoding acceptance metrics (MTP/EAGLE3 draft runs) — from the server log before teardown.
# llama-server logs draft stats e.g. "n_draft", "n_accept", "accept = X%".
if printf '%s\n' "${EXTRA[@]:-}" | grep -q "spec-type\|model-draft"; then
  echo "==> SPEC-METRICS (acceptance):"
  docker logs "$NAME" 2>&1 | grep -iE "n_accept|n_draft|accept|draft" | tail -20 || true
fi
