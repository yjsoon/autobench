#!/usr/bin/env bash
# Run the standard benchmark sweep (llama-bench + ShareGPT serving c1/c8/c32)
# for ONE GGUF, with the same logging/summary format as overnight-llamacpp-local.sh.
#
# Usage: sweep-gguf.sh <gguf-path-under-MODELS_DIR> <label-prefix> [conc list, default "1 8 32"]
# Outputs live under results/sweep-<label-prefix>-<timestamp>/.
set -u -o pipefail
cd "$(dirname "$0")/.."

FILE="${1:?need a gguf path under MODELS_DIR}"
PREFIX="${2:?need a label prefix, e.g. gptoss-20b-mxfp4}"
CONCS="${3:-1 8 32}"

STAMP="$(date '+%Y%m%d-%H%M%S')"
OUTDIR="${OUTDIR:-results/sweep-$PREFIX-$STAMP}"
mkdir -p "$OUTDIR"

SUMMARY="$OUTDIR/summary.tsv"
RUNLOG="$OUTDIR/run.log"
VRAM_DIR="/sys/class/drm/card0/device"

LLAMA_IMAGE="${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:full-vulkan}"
IMAGE_DIGEST="$(docker inspect --format '{{index .RepoDigests 0}}' "$LLAMA_IMAGE" 2>/dev/null || echo "$LLAMA_IMAGE")"

{
  echo "started_at	$(date '+%Y-%m-%d %H:%M:%S %z')"
  echo "llamacpp_image	$IMAGE_DIGEST"
  echo "outdir	$OUTDIR"
} > "$RUNLOG"
printf 'started_at\tfinished_at\tstatus\tlabel\tcommand\tresult\tmem\tvram_peak_gb\tvram_delta_gb\n' > "$SUMMARY"

bytes_to_gib() {
  awk -v b="${1:-0}" 'BEGIN { printf "%.2f", b / 1024 / 1024 / 1024 }'
}

read_vram() {
  if [ -r "$VRAM_DIR/mem_info_vram_used" ]; then
    cat "$VRAM_DIR/mem_info_vram_used"
  else
    echo 0
  fi
}

max_vram_from_samples() {
  awk 'NR > 1 && $2 > max { max = $2 } END { print max + 0 }' "$1"
}

run_one() {
  label="$1"
  shift
  logfile="$OUTDIR/$label.log"
  vramlog="$OUTDIR/$label.vram.tsv"
  start_human="$(date '+%Y-%m-%d %H:%M:%S %z')"
  base_vram="$(read_vram)"
  cmd="$*"

  echo "[$start_human] START $label :: $cmd" | tee -a "$RUNLOG"
  printf 'epoch\tvram_used_bytes\n' > "$vramlog"
  (
    while :; do
      printf '%s\t%s\n' "$(date '+%s')" "$(read_vram)" >> "$vramlog"
      sleep 10
    done
  ) &
  sampler=$!

  "$@" > "$logfile" 2>&1
  status=$?

  kill "$sampler" >/dev/null 2>&1 || true
  wait "$sampler" 2>/dev/null || true

  end_human="$(date '+%Y-%m-%d %H:%M:%S %z')"
  max_vram="$(max_vram_from_samples "$vramlog")"
  vram_peak_gb="$(bytes_to_gib "$max_vram")"
  vram_delta_gb="$(bytes_to_gib "$((max_vram - base_vram))")"
  result="$(grep '^RESULT ' "$logfile" | tail -1 | tr '\t' ' ' || true)"
  mem="$(grep -E '^MEM |  mem \\(system MemAvailable delta\\)' "$logfile" | tail -1 | tr '\t' ' ' || true)"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$start_human" "$end_human" "$status" "$label" "$cmd" "$result" "$mem" \
    "$vram_peak_gb" "$vram_delta_gb" >> "$SUMMARY"

  echo "[$end_human] END $label status=$status result=${result:-none} vram_peak=${vram_peak_gb}GiB delta=${vram_delta_gb}GiB" \
    | tee -a "$RUNLOG"
  return 0
}

run_one "$PREFIX-llamabench-pp512-tg128" \
  scripts/bench-llamacpp.sh "$FILE" 512 128 99

for conc in $CONCS; do
  run_one "$PREFIX-serving-c${conc}" \
    scripts/bench-llamacpp-serving.sh "$FILE" 65536 "$conc" 1000 900 256 99
done

echo "finished_at	$(date '+%Y-%m-%d %H:%M:%S %z')" >> "$RUNLOG"
echo "DONE: $OUTDIR"
