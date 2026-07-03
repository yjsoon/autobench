#!/usr/bin/env bash
# Benchmark one GGUF with llama.cpp (llama-bench) and measure peak memory.
#
# Reports prefill (pp) and decode (tg) tok/s, and peak memory via the container's
# cgroup-v2 memory.peak (primary) plus a 10s-sampled docker-stats max and the system
# MemAvailable delta (cross-checks). Strix Halo is unified memory: no discrete-GPU
# meter; cross-check with /sys/class/drm/card*/device/mem_info_vram_used.
#
# Strix Halo port: uses the Vulkan (RADV) llama.cpp image with /dev/dri + /dev/kfd
# passed through instead of --gpus all. Override MODELS_DIR / LLAMACPP_IMAGE via env.
#
# Usage: scripts/bench-llamacpp.sh <model.gguf-under-$MODELS_DIR> [pp] [tg] [ngl]
#   e.g. scripts/bench-llamacpp.sh lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf 512 128 99
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-$HOME/.lmstudio/models}"
IMAGE="${LLAMACPP_IMAGE:-ghcr.io/ggml-org/llama.cpp:full-vulkan}"
FILE="${1:?need a gguf filename under $MODELS_DIR}"
PP="${2:-512}"; TG="${3:-128}"; NGL="${4:-99}"
NAME="bench-$(echo "$FILE" | tr -c 'A-Za-z0-9' '-')"

mib() { awk '{printf "%.1f", $1/1024/1024}'; }            # bytes -> MiB
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }          # kB -> GiB

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)   # kB
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "==> llama-bench $FILE  (pp=$PP tg=$TG ngl=$NGL)"
# NB: this image uses a dispatcher entrypoint — llama-bench is invoked via --bench.
docker run -d --name "$NAME" --device /dev/dri --device /dev/kfd \
  -v "$MODELS_DIR":/models:ro "$IMAGE" \
  --bench -m "/models/$FILE" -p "$PP" -n "$TG" -ngl "$NGL" >/dev/null

cid=$(docker inspect -f '{{.Id}}' "$NAME")
# Locate the container's cgroup memory.peak (systemd or cgroupfs driver).
peak_file=""
for p in "/sys/fs/cgroup/system.slice/docker-$cid.scope/memory.peak" \
         "/sys/fs/cgroup/docker/$cid/memory.peak"; do
  [ -r "$p" ] && peak_file="$p" && break
done

stats_max_mib=0; min_avail=$base_avail
while [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" = "true" ]; do
  mem=$(docker stats --no-stream --format '{{.MemUsage}}' "$NAME" 2>/dev/null | awk '{print $1}')
  cur_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  [ -n "$mem" ] && echo "    [sample] container=$mem  MemAvailable=$(echo "$cur_avail" | gib_kb)GiB"
  [ "$cur_avail" -lt "$min_avail" ] && min_avail=$cur_avail
  sleep 10
done

docker wait "$NAME" >/dev/null 2>&1 || true
log=$(docker logs "$NAME" 2>&1)

peak_bytes=""; [ -n "$peak_file" ] && peak_bytes=$(cat "$peak_file" 2>/dev/null || true)
docker rm -f "$NAME" >/dev/null 2>&1 || true

echo "----- llama-bench output -----"
echo "$log" | grep -E '\| *(pp|tg)[0-9]+ *\|' || echo "$log" | tail -20
echo "------------------------------"

# Table rows look like: | ...model... | pp512 |   3362.66 ± 691.07 |  — grab the mean t/s.
pp_ts=$(echo "$log" | grep -oE "\| *pp$PP *\| *[0-9.]+" | grep -oE '[0-9.]+$' | tail -1)
tg_ts=$(echo "$log" | grep -oE "\| *tg$TG *\| *[0-9.]+" | grep -oE '[0-9.]+$' | tail -1)
sys_delta_gib=$(echo "$((base_avail - min_avail))" | gib_kb)

echo "RESULT file=$FILE"
echo "  prefill_toks (pp$PP) = ${pp_ts:-?} tok/s"
echo "  decode_toks  (tg$TG) = ${tg_ts:-?} tok/s"
[ -n "$peak_bytes" ] && echo "  mem_peak (cgroup memory.peak) = $(echo "$peak_bytes" | mib) MiB"
echo "  mem (system MemAvailable delta) = ${sys_delta_gib} GiB"
