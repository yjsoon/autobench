#!/usr/bin/env bash
# Serving benchmark against an ALREADY-RUNNING LM Studio server (Strix Halo addition).
#
# LM Studio (Vulkan runtime) exposes an OpenAI-compatible API, so scripts/bench-serving.py
# drives it exactly like llama.cpp / vLLM / SGLang. Nothing is launched or torn down here:
# start the server first (LM Studio GUI -> Developer -> Start Server, or `lms server start`),
# load the model with GPU offload set to all layers, then run this.
#
# Headline memory = system MemAvailable delta from idle (10s sampling), as on the other
# wrappers. Cross-check GPU-side with /sys/class/drm/card*/device/mem_info_vram_used.
#
# Usage: bench-lmstudio-serving.sh <model-id> [conc] [num_prompts] [max_seconds] [max_tokens]
#   e.g. bench-lmstudio-serving.sh qwen3.6-35b-a3b 1 1000 900 256
#   Model id must match what the server reports at $BASE_URL/v1/models.
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
BASE_URL="${LMSTUDIO_URL:-http://localhost:1234}"

MODEL="${1:?need a model id as served by LM Studio (see $BASE_URL/v1/models)}"
CONC="${2:-1}"; NUMP="${3:-1000}"; MAXS="${4:-900}"; MAXTOK="${5:-256}"
NAME="lms-$(echo "$MODEL" | tr -c 'A-Za-z0-9' '-')"
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }

curl -fsS "$BASE_URL/v1/models" >/dev/null 2>&1 || {
  echo "!! no LM Studio server at $BASE_URL — start it and load '$MODEL' first"; exit 1; }

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
min_avail=$base_avail
( while :; do
    cur=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$cur" -lt "$min_avail" ] && min_avail=$cur
    echo "$min_avail" > /tmp/$NAME.minavail; sleep 10
  done ) &
sampler=$!
trap 'kill "$sampler" >/dev/null 2>&1 || true' EXIT

echo "==> run ShareGPT serving benchmark against LM Studio (conc=$CONC num=$NUMP max=${MAXS}s)"
python3 "$BENCH" --base-url "$BASE_URL" --model "$MODEL" \
  --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" \
  --concurrency "$CONC" --max-tokens "$MAXTOK" || true

kill "$sampler" >/dev/null 2>&1 || true
min_avail=$(cat /tmp/$NAME.minavail 2>/dev/null || echo "$base_avail"); rm -f /tmp/$NAME.minavail
echo "MEM mem_gb=$(echo "$((base_avail - min_avail))" | gib_kb) mem_source=\"system MemAvailable delta (10s sampling); NB baseline includes the pre-loaded model\""
