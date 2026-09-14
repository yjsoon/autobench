#!/usr/bin/env bash
set -u

container=""
floor_gb=""
interval=2
log_path=""

while (($#)); do
  case "$1" in
    --container) container=${2:?missing container}; shift 2 ;;
    --floor-gb) floor_gb=${2:?missing floor}; shift 2 ;;
    --interval) interval=${2:?missing interval}; shift 2 ;;
    --log) log_path=${2:?missing log}; shift 2 ;;
    *) echo "usage: $0 --container NAME --floor-gb N [--interval SEC] --log PATH" >&2; exit 2 ;;
  esac
done

if [[ -z "$container" || -z "$floor_gb" || -z "$log_path" ]]; then
  echo "container, floor-gb, and log are required" >&2
  exit 2
fi

floor_kb=$((floor_gb * 1024 * 1024))
mkdir -p "$(dirname "$log_path")"
exec >>"$log_path" 2>&1
echo "watchdog_start container=$container floor_gb=$floor_gb interval_s=$interval"

while :; do
  running=$(docker inspect -f '{{.State.Running}}' "$container" 2>/dev/null || true)
  [[ "$running" == "true" ]] || break
  available_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  echo "$(date '+%Y-%m-%d %H:%M:%S %z') available_kb=$available_kb floor_kb=$floor_kb"
  if [[ -n "$available_kb" && "$available_kb" -lt "$floor_kb" ]]; then
    echo "WATCHDOG_TRIP available_kb=$available_kb floor_kb=$floor_kb action=docker_kill"
    docker kill --signal TERM "$container" || true
    exit 0
  fi
  sleep "$interval"
done

echo "watchdog_exit container_not_running"
