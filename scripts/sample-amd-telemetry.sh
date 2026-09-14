#!/usr/bin/env bash
set -u

container=""
output=""
interval=2

while (($#)); do
  case "$1" in
    --container) container=${2:?missing container}; shift 2 ;;
    --output) output=${2:?missing output}; shift 2 ;;
    --interval) interval=${2:?missing interval}; shift 2 ;;
    *) echo "usage: $0 --container NAME --output PATH [--interval SEC]" >&2; exit 2 ;;
  esac
done

if [[ -z "$container" || -z "$output" ]]; then
  echo "container and output are required" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"
{
  echo "timestamp,mem_available_kb,vram_used_bytes,gpu_busy_percent,temp_mC,power_uW,sclk_hz"
  while :; do
    running=$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)
    [[ "$running" == "true" ]] || break
    ts=$(date '+%Y-%m-%d %H:%M:%S %z')
    mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    vram=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo "")
    busy=$(cat /sys/class/drm/card1/device/gpu_busy_percent 2>/dev/null || echo "")
    temp=$(for f in /sys/class/drm/card1/device/hwmon/hwmon*/temp1_input; do [[ -r "$f" ]] && { cat "$f"; break; }; done)
    power=$(for f in /sys/class/drm/card1/device/hwmon/hwmon*/power1_input; do [[ -r "$f" ]] && { cat "$f"; break; }; done)
    sclk=$(cat /sys/class/drm/card1/device/freq1_input 2>/dev/null || echo "")
    echo "$ts,$mem,$vram,$busy,${temp:-},${power:-},${sclk:-}"
    sleep "$interval"
  done
} >"$output"
