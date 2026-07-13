#!/usr/bin/env bash
# SGLang serving sweep for one HF model on Strix Halo (gfx1151): launch the
# server ONCE, then drive bench-serving.py at several concurrencies against it,
# sampling GPU VRAM (sysfs) throughout. Loads the model once (unlike calling
# bench-sglang-serving.sh per concurrency, which reloads each time).
#
# Usage: sweep-sglang.sh <hf-model> <label-prefix> [ctx] [concs] [extra sglang args...]
#   e.g. sweep-sglang.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit qwen35-a3b-awq 8192 "1 8 32" \
#          --mem-fraction-static 0.5 --max-total-tokens 32768 --max-mamba-cache-size 64 \
#          --attention-backend triton --disable-cuda-graph
set -u -o pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null || true; set +a

MODEL="${1:?need an HF model path}"
PREFIX="${2:?need a label prefix}"
CTX="${3:-8192}"
CONCS="${4:-1 8 32}"
shift 4 2>/dev/null || shift $#
EXTRA=("$@")

IMAGE="${SGLANG_IMAGE:-strix-halo-sglang:dev}"
PORT=30000
NUMP="${NUMP:-1000}"; MAXS="${MAXS:-900}"; MAXTOK="${MAXTOK:-256}"
DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
VRAM_FILE="/sys/class/drm/card0/device/mem_info_vram_used"
NAME="sgl-sweep"

STAMP="$(date '+%Y%m%d-%H%M%S')"
OUT="results/sglang-$PREFIX-$STAMP"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf 'label\tconc\tresult\tmem\tvram_base_gb\tvram_peak_gb\tvram_delta_gb\n' > "$SUMMARY"

gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }
gib_b()  { awk -v b="${1:-0}" 'BEGIN{printf "%.2f", b/1073741824}'; }
read_vram() { [ -r "$VRAM_FILE" ] && cat "$VRAM_FILE" || echo 0; }

IMAGE_DIGEST="$(docker inspect --format '{{index .Id}}' "$IMAGE" 2>/dev/null || echo "$IMAGE")"
{ echo "model	$MODEL"; echo "image	$IMAGE ($IMAGE_DIGEST)"; echo "ctx	$CTX";
  echo "extra	${EXTRA[*]:-}"; echo "started	$(date '+%Y-%m-%d %H:%M:%S %z')"; } > "$OUT/run.log"

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "==> launch SGLang $MODEL (ctx=$CTX) extra=[${EXTRA[*]:-}]" | tee -a "$OUT/run.log"
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri \
  --ipc=host --shm-size 32g --security-opt seccomp=unconfined -p "$PORT:$PORT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/strix-halo-sglang-tunableop:/root/.tunableop" \
  --env "HF_TOKEN=${HF_TOKEN:-}" --env "SGLANG_FORCE_NATIVE_LAYERNORM=1" \
  "$IMAGE" python3 -m sglang.launch_server --model-path "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --context-length "$CTX" "${EXTRA[@]}" >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> waiting for /health (download+load can be many minutes) ..." | tee -a "$OUT/run.log"
HT="${HEALTH_TIMEOUT:-1800}"
for i in $(seq 1 "$HT"); do
  curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "    ready after ${i}s" | tee -a "$OUT/run.log"; break; }
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! container exited during load:" | tee -a "$OUT/run.log"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$OUT/run.log"; exit 1
  fi
  sleep 1
  [ "$i" = "$HT" ] && { echo "!! health timeout" | tee -a "$OUT/run.log"; docker logs "$NAME" 2>&1 | tail -40 | tee -a "$OUT/run.log"; exit 1; }
done

vram_base="$(read_vram)"
echo "vram_base_gb=$(gib_b "$vram_base")" | tee -a "$OUT/run.log"

for conc in $CONCS; do
  echo "==> [$(date '+%H:%M:%S')] serving conc=$conc" | tee -a "$OUT/run.log"
  vlog="$OUT/$PREFIX-c$conc.vram.tsv"; printf 'epoch\tvram\n' > "$vlog"
  ( while :; do printf '%s\t%s\n' "$(date +%s)" "$(read_vram)" >> "$vlog"; sleep 5; done ) & sampler=$!
  base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  python3 "$BENCH" --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" \
    --concurrency "$conc" --max-tokens "$MAXTOK" > "$OUT/$PREFIX-c$conc.log" 2>&1 || true
  kill "$sampler" >/dev/null 2>&1 || true; wait "$sampler" 2>/dev/null || true
  min_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  peak=$(awk 'NR>1&&$2>m{m=$2}END{print m+0}' "$vlog")
  result="$(grep '^RESULT ' "$OUT/$PREFIX-c$conc.log" | tail -1 | tr '\t' ' ')"
  mem="mem_gb=$(echo "$((base_avail - min_avail))" | gib_kb)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$PREFIX" "$conc" "$result" "$mem" \
    "$(gib_b "$vram_base")" "$(gib_b "$peak")" "$(gib_b "$((peak - vram_base))")" >> "$SUMMARY"
  echo "   c$conc: ${result:-NO RESULT} | vram_peak=$(gib_b "$peak") delta=$(gib_b "$((peak - vram_base))")" | tee -a "$OUT/run.log"
done

echo "finished	$(date '+%Y-%m-%d %H:%M:%S %z')" >> "$OUT/run.log"
echo "DONE: $OUT"; column -t "$SUMMARY"