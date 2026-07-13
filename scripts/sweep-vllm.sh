#!/usr/bin/env bash
# vLLM serving sweep for one HF model on Strix Halo (gfx1151): launch the server
# ONCE (--max-num-seqs = max concurrency), then drive bench-serving.py at several
# client concurrencies, sampling GPU VRAM. gfx1151 needs --enforce-eager (vLLM's
# CUDA-graph / torch.compile capture GPU-faults during profile_run otherwise).
#
# Usage: sweep-vllm.sh <hf-model> <label-prefix> [ctx] [concs] [extra vllm serve args...]
#   e.g. sweep-vllm.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit qwen35-a3b-awq-vllm 8192 "1 8 32"
set -u -o pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null || true; set +a

MODEL="${1:?need an HF model path}"
PREFIX="${2:?need a label prefix}"
CTX="${3:-8192}"
CONCS="${4:-1 8 32}"
shift 4 2>/dev/null || shift $#
EXTRA=("$@")

IMAGE="${VLLM_IMAGE:-kyuz0/vllm-therock-gfx1151:stable}"
PORT=8000
NUMP="${NUMP:-1000}"; MAXS="${MAXS:-900}"; MAXTOK="${MAXTOK:-256}"
MAXSEQS="${MAXSEQS:-32}"          # server batch cap; >= max client concurrency
GPU_UTIL="${GPU_UTIL:-0.5}"       # keep low: OpenCode server co-resides in the UMA pool
DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
VRAM_FILE="/sys/class/drm/card0/device/mem_info_vram_used"
NAME="vllm-sweep"

STAMP="$(date '+%Y%m%d-%H%M%S')"
OUT="results/vllm-$PREFIX-$STAMP"
mkdir -p "$OUT"
SUMMARY="$OUT/summary.tsv"
printf 'label\tconc\tresult\tvram_base_gb\tvram_peak_gb\tvram_delta_gb\n' > "$SUMMARY"

gib_b() { awk -v b="${1:-0}" 'BEGIN{printf "%.2f", b/1073741824}'; }
read_vram() { [ -r "$VRAM_FILE" ] && cat "$VRAM_FILE" || echo 0; }

{ echo "model	$MODEL"; echo "image	$IMAGE"; echo "ctx	$CTX  max_num_seqs	$MAXSEQS  gpu_util	$GPU_UTIL";
  echo "extra	${EXTRA[*]:-}"; echo "started	$(date '+%Y-%m-%d %H:%M:%S %z')"; } > "$OUT/run.log"

docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "==> launch vLLM $MODEL (ctx=$CTX, --enforce-eager, max-num-seqs=$MAXSEQS)" | tee -a "$OUT/run.log"
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri \
  --ipc=host --security-opt seccomp=unconfined -p "$PORT:$PORT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  --env "HF_TOKEN=${HF_TOKEN:-}" --entrypoint vllm \
  "$IMAGE" serve "$MODEL" --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$CTX" --gpu-memory-utilization "$GPU_UTIL" --max-num-seqs "$MAXSEQS" \
  --enforce-eager "${EXTRA[@]}" >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> waiting for /health (compile+load can be minutes) ..." | tee -a "$OUT/run.log"
HT="${HEALTH_TIMEOUT:-1200}"
for i in $(seq 1 "$HT"); do
  curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1 && { echo "    ready after ${i}s" | tee -a "$OUT/run.log"; break; }
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! container exited during load:" | tee -a "$OUT/run.log"; docker logs "$NAME" 2>&1 | tail -50 | tee -a "$OUT/run.log"; exit 1
  fi
  sleep 1
  [ "$i" = "$HT" ] && { echo "!! health timeout" | tee -a "$OUT/run.log"; docker logs "$NAME" 2>&1 | tail -50 | tee -a "$OUT/run.log"; exit 1; }
done

vram_base="$(read_vram)"
echo "vram_base_gb=$(gib_b "$vram_base")" | tee -a "$OUT/run.log"

for conc in $CONCS; do
  echo "==> [$(date '+%H:%M:%S')] serving conc=$conc" | tee -a "$OUT/run.log"
  vlog="$OUT/$PREFIX-c$conc.vram.tsv"; printf 'epoch\tvram\n' > "$vlog"
  ( while :; do printf '%s\t%s\n' "$(date +%s)" "$(read_vram)" >> "$vlog"; sleep 5; done ) & sampler=$!
  python3 "$BENCH" --base-url "http://localhost:$PORT" --model "$MODEL" \
    --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" \
    --concurrency "$conc" --max-tokens "$MAXTOK" > "$OUT/$PREFIX-c$conc.log" 2>&1 || true
  kill "$sampler" >/dev/null 2>&1 || true; wait "$sampler" 2>/dev/null || true
  peak=$(awk 'NR>1&&$2>m{m=$2}END{print m+0}' "$vlog")
  result="$(grep '^RESULT ' "$OUT/$PREFIX-c$conc.log" | tail -1 | tr '\t' ' ')"
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$PREFIX" "$conc" "$result" \
    "$(gib_b "$vram_base")" "$(gib_b "$peak")" "$(gib_b "$((peak - vram_base))")" >> "$SUMMARY"
  echo "   c$conc: ${result:-NO RESULT} | vram_peak=$(gib_b "$peak")" | tee -a "$OUT/run.log"
done

echo "finished	$(date '+%Y-%m-%d %H:%M:%S %z')" >> "$OUT/run.log"
echo "DONE: $OUT"; column -t "$SUMMARY"