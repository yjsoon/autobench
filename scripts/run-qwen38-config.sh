#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ "$#" -lt 9 || "$#" -gt 10 ]]; then
  echo "usage: $0 SLUG ENGINE QUANT SPEC CONTEXT CONCURRENCY NUM_PROMPTS MAX_SECONDS MAX_TOKENS [RECIPE_VARIANT]" >&2
  exit 2
fi

SLUG=$1
ENGINE=$2
QUANT=$3
SPEC=$4
CTX=$5
CONC=$6
NUMP=$7
MAXS=$8
MAXTOK=$9
RECIPE_VARIANT=${10:-legacy}
ROOT=$(pwd)
DATASET="$ROOT/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$ROOT/scripts/bench-serving.py"
PROBE="$ROOT/scripts/probe-serving.py"
RESULT_DIR="$ROOT/results"
LOG_DIR="$ROOT/logs"
mkdir -p "$RESULT_DIR" "$LOG_DIR" "$HOME/.cache/llama.cpp" "$HOME/.cache/strix-halo-sglang-tunableop"
if [[ "$#" -eq 10 ]]; then
  printf '%s\n' "scripts/run-qwen38-config.sh $SLUG $ENGINE $QUANT $SPEC $CTX $CONC $NUMP $MAXS $MAXTOK $RECIPE_VARIANT" >> "$ROOT/commands.log"
else
  printf '%s\n' "scripts/run-qwen38-config.sh $SLUG $ENGINE $QUANT $SPEC $CTX $CONC $NUMP $MAXS $MAXTOK" >> "$ROOT/commands.log"
fi
log_command() {
  local label=$1
  shift
  {
    printf '%s' "$label"
    printf ' %q' "$@"
    printf '\n'
  } >> "$ROOT/commands.log"
}

WATCHDOG_FLOOR=${QWEN38_WATCHDOG_FLOOR_GB:-8}
WATCHDOG_INTERVAL=${QWEN38_WATCHDOG_INTERVAL_S:-2}
HEALTH_TIMEOUT=${QWEN38_HEALTH_TIMEOUT_S:-1800}
MODEL_ID="Qwen/Qwen3.8-27B"
MODEL_REF=""
MODEL_REV=""
IMAGE=""
PORT=""
NAME=""
SERVER_LOG=""
IMAGE_LOG=""
WARMUP_LOG=""
BENCH_LOG=""
TELEMETRY_LOG=""
WATCHDOG_LOG=""
SPEC_LOG=""
TELEMETRY_PID=""
WATCHDOG_PID=""
BASE_AVAIL=""
BASE_VRAM=""
CONTAINER_ID=""

cleanup() {
  if [[ -n "$NAME" ]]; then
    docker logs "$NAME" >"$SERVER_LOG" 2>&1 || true
  fi
  if [[ -n "$TELEMETRY_PID" ]]; then
    kill "$TELEMETRY_PID" >/dev/null 2>&1 || true
    wait "$TELEMETRY_PID" >/dev/null 2>&1 || true
    TELEMETRY_PID=""
  fi
  if [[ -n "$WATCHDOG_PID" ]]; then
    kill "$WATCHDOG_PID" >/dev/null 2>&1 || true
    wait "$WATCHDOG_PID" >/dev/null 2>&1 || true
    WATCHDOG_PID=""
  fi
  if [[ -n "$NAME" ]]; then
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    NAME=""
  fi
}
trap cleanup EXIT
trap 'cleanup; exit 143' INT TERM

select_model() {
  case "$ENGINE:$QUANT" in
    llama:q4_k_m)
      IMAGE="ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41"
      MODEL_REF="unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M"
      MODEL_REV="4ca720788d1e01f1bff70c033e0d0028fd02e502"
      PORT=8081
      ;;
    llama:q8_0)
      IMAGE="ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41"
      MODEL_REF="unsloth/Qwen3.8-27B-GGUF:Q8_0"
      MODEL_REV="4ca720788d1e01f1bff70c033e0d0028fd02e502"
      PORT=8081
      ;;
    llama:bf16)
      IMAGE="ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41"
      MODEL_REF="unsloth/Qwen3.8-27B-GGUF:BF16"
      MODEL_REV="4ca720788d1e01f1bff70c033e0d0028fd02e502"
      PORT=8081
      ;;
    sglang:bf16)
      IMAGE="strix-halo-sglang:dev@sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272"
      MODEL_REF="Qwen/Qwen3.8-27B"
      MODEL_REV="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
      PORT=30000
      ;;
    sglang:fp8)
      IMAGE="strix-halo-sglang:dev@sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272"
      MODEL_REF="Qwen/Qwen3.8-27B-FP8"
      MODEL_REV="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
      PORT=30000
      ;;
    sglang:nvfp4)
      IMAGE="strix-halo-sglang:dev@sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272"
      MODEL_REF="unsloth/Qwen3.8-27B-NVFP4"
      MODEL_REV="9e3d73c76eddb75f795cc24ccfbc5affe41c66bd"
      PORT=30000
      ;;
    vllm:bf16)
      IMAGE="kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2"
      MODEL_REF="Qwen/Qwen3.8-27B"
      MODEL_REV="1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
      PORT=8000
      ;;
    vllm:fp8)
      IMAGE="kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2"
      MODEL_REF="Qwen/Qwen3.8-27B-FP8"
      MODEL_REV="017b9c7af6b5689d5dd426a76e0bc077eb5ca20a"
      PORT=8000
      ;;
    vllm:nvfp4)
      IMAGE="kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2"
      MODEL_REF="unsloth/Qwen3.8-27B-NVFP4"
      MODEL_REV="9e3d73c76eddb75f795cc24ccfbc5affe41c66bd"
      PORT=8000
      ;;
    vllm:quark_int4)
      IMAGE="kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2"
      MODEL_REF="amd/Qwen3.8-27B-Quark-AWQ-INT4-W4A16"
      MODEL_REV="e4e5466bd950f15f221f474ceb0aa6f70056094d"
      PORT=8000
      ;;
    vllm:quark_mxfp4)
      IMAGE="kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2"
      MODEL_REF="amd/Qwen3.8-27B-Quark-AWQ-MXFP4"
      MODEL_REV="5233554c5fa56afda40150556b95573c2d7d29c0"
      PORT=8000
      ;;
    *)
      echo "unsupported engine/quant: $ENGINE/$QUANT" >&2
      exit 2
      ;;
  esac
}

run_repeat() {
  local repeat=$1
  NAME="qwen38-$SLUG-r$repeat"
  SERVER_LOG="$LOG_DIR/server-$SLUG-r$repeat.log"
  IMAGE_LOG="$LOG_DIR/image-$SLUG-r$repeat.log"
  WARMUP_LOG="$LOG_DIR/coherence-$SLUG-r$repeat.json"
  BENCH_LOG="$LOG_DIR/bench-$SLUG-r$repeat.log"
  TELEMETRY_LOG="$LOG_DIR/accel-$SLUG-r$repeat.csv"
  WATCHDOG_LOG="$LOG_DIR/mem-watchdog-$SLUG-r$repeat.log"
  SPEC_LOG="$LOG_DIR/spec-$SLUG-r$repeat.log"
  TELEMETRY_PID=""
  WATCHDOG_PID=""
  BASE_AVAIL=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
  BASE_VRAM=$(cat /sys/class/drm/card1/device/mem_info_vram_used 2>/dev/null || echo "")
  printf '%s\n' "repeat=$repeat baseline_mem_available_kb=$BASE_AVAIL baseline_vram_used_bytes=$BASE_VRAM" >"$IMAGE_LOG"
  printf '%s\n' "engine=$ENGINE quant=$QUANT speculation=$SPEC recipe_variant=$RECIPE_VARIANT context=$CTX concurrency=$CONC model_ref=$MODEL_REF model_revision=$MODEL_REV" >>"$IMAGE_LOG"
  docker image inspect "$IMAGE" >>"$IMAGE_LOG" 2>&1 || true

  local docker_cmd
  docker_cmd=(docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri --ipc=host --shm-size 32g --security-opt seccomp=unconfined -p "$PORT:$PORT")
  case "$ENGINE" in
    llama)
      docker_cmd+=(-v "$HOME/.cache/huggingface:/root/.cache/huggingface")
      # llama-server's -c is the total context pool. Keep it fixed across the
      # sweep, as in autobench's existing llama.cpp pages; --parallel divides
      # that pool into slots.
      docker_cmd+=("$IMAGE" --server --hf-repo "$MODEL_REF" --no-mmproj -ngl 99 -c "$CTX" --parallel "$CONC" -cb --reasoning off --host 0.0.0.0 --port "$PORT" --metrics --log-timestamps)
      if [[ "$RECIPE_VARIANT" == "q4kv" ]]; then
        # qwen38-mtp recipe: quantized K/V cache and forced flash attention.
        docker_cmd+=(--cache-type-k q4_0 --cache-type-v q4_0 -fa 1)
      elif [[ "$RECIPE_VARIANT" != "legacy" ]]; then
        echo "unsupported llama recipe variant: $RECIPE_VARIANT" >&2
        return 2
      fi
      case "$SPEC" in
        mtp|mtp2|mtp3|mtp4)
          # With draft-mtp, this build discovers the repository's MTP sidecar
          # from the primary HF repo. Do not substitute the ordinary Q4_0
          # target file as a draft model.
          local mtp_n=3
          case "$SPEC" in
            mtp2) mtp_n=2 ;;
            mtp3) mtp_n=3 ;;
            mtp4) mtp_n=4 ;;
          esac
          docker_cmd+=(--spec-type draft-mtp --spec-draft-n-max "$mtp_n")
          ;;
        ngram)
          docker_cmd+=(--spec-type ngram-simple --spec-ngram-simple-size-n 12 --spec-ngram-simple-size-m 48)
          ;;
        base)
          ;;
        *)
          echo "unsupported llama speculation: $SPEC" >&2
          return 2
          ;;
      esac
      ;;
    sglang)
      docker_cmd+=(-v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/strix-halo-sglang-tunableop:/root/.tunableop")
      docker_cmd+=(--env SGLANG_FORCE_NATIVE_LAYERNORM=1 --env PYTORCH_ROCM_ARCH=gfx1151 --env PYTORCH_TUNABLEOP_ENABLED=1)
      docker_cmd+=("$IMAGE" python3 -m sglang.launch_server --model-path "$MODEL_REF" --revision "$MODEL_REV" --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL_ID" --context-length "$CTX" --max-running-requests "$CONC" --mem-fraction-static 0.80 --attention-backend triton --linear-attn-backend triton --mamba-backend triton --disable-cuda-graph --enable-metrics)
      case "$SPEC" in
        mtp)
          docker_cmd+=(--speculative-algorithm NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 3)
          ;;
        base)
          ;;
        *)
          echo "unsupported sglang speculation: $SPEC" >&2
          return 2
          ;;
      esac
      ;;
    vllm)
      docker_cmd+=(-v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm")
      docker_cmd+=(--env HSA_OVERRIDE_GFX_VERSION=11.5.1)
      docker_cmd+=("$IMAGE" vllm serve "$MODEL_REF" --revision "$MODEL_REV" --host 0.0.0.0 --port "$PORT" --served-model-name "$MODEL_ID" --max-model-len "$CTX" --gpu-memory-utilization 0.80 --max-num-seqs "$CONC" --enforce-eager --trust-remote-code)
      case "$SPEC" in
        mtp)
          docker_cmd+=(--speculative-config '{"method":"mtp","num_speculative_tokens":3}')
          ;;
        base)
          ;;
        *)
          echo "unsupported vllm speculation: $SPEC" >&2
          return 2
          ;;
      esac
      ;;
  esac

  log_command "docker_run slug=$SLUG repeat=$repeat" "${docker_cmd[@]}"
  CONTAINER_ID=$("${docker_cmd[@]}" 2>>"$IMAGE_LOG")
  printf '%s\n' "container_id=$CONTAINER_ID" >>"$IMAGE_LOG"
  scripts/sample-amd-telemetry.sh --container "$NAME" --output "$TELEMETRY_LOG" --interval 2 &
  TELEMETRY_PID=$!
  : >"$WATCHDOG_LOG"
  scripts/mem-watchdog.sh --container "$NAME" --floor-gb "$WATCHDOG_FLOOR" --interval "$WATCHDOG_INTERVAL" --log "$WATCHDOG_LOG" &
  WATCHDOG_PID=$!

  local ready=0
  local i
  for i in $(seq 1 "$HEALTH_TIMEOUT"); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      ready=1
      printf '%s\n' "ready_after_s=$i" >>"$IMAGE_LOG"
      break
    fi
    if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || true)" != "true" ]]; then
      printf '%s\n' "container_exited_during_health=1" >>"$IMAGE_LOG"
      break
    fi
    sleep 1
  done
  if [[ "$ready" != 1 ]]; then
    docker logs "$NAME" >"$SERVER_LOG" 2>&1 || true
    echo "server did not become healthy for $SLUG repeat $repeat" >&2
    return 1
  fi

  set +e
  python3 "$PROBE" --base-url "http://127.0.0.1:$PORT" --model "$MODEL_ID" --dataset "$DATASET" --max-tokens 64 --output "$WARMUP_LOG" >"$LOG_DIR/probe-$SLUG-r$repeat.log" 2>&1
  local probe_status=$?
  set -e
  if [[ "$probe_status" != 0 ]]; then
    docker logs "$NAME" >"$SERVER_LOG" 2>&1 || true
    echo "warmup/coherence probe failed for $SLUG repeat $repeat" >&2
    return 1
  fi

  local result_file="$RESULT_DIR/$SLUG.repeat$repeat.json"
  local samples_file="$RESULT_DIR/$SLUG.repeat$repeat.samples.json"
  local bench_cmd
  bench_cmd=(python3 "$BENCH" --base-url "http://127.0.0.1:$PORT" --model "$MODEL_ID" --dataset "$DATASET" --num-prompts "$NUMP" --max-seconds "$MAXS" --concurrency "$CONC" --max-tokens "$MAXTOK" --request-timeout 120 --slo-ttft-ms 2000 --slo-tpot-ms 50 --output "$result_file" --samples-output "$samples_file")
  log_command "benchmark slug=$SLUG repeat=$repeat" "${bench_cmd[@]}"
  set +e
  "${bench_cmd[@]}" >"$BENCH_LOG" 2>&1
  local bench_status=$?
  set -e
  docker logs "$NAME" >"$SERVER_LOG" 2>&1 || true
  if [[ "$SPEC" != "base" ]]; then
    grep -iE 'accept|spec|draft|n_accept|n_draft' "$SERVER_LOG" >"$SPEC_LOG" || true
  else
    printf '%s\n' "not_speculative" >"$SPEC_LOG"
  fi
  if grep -q 'WATCHDOG_TRIP' "$WATCHDOG_LOG" 2>/dev/null; then
    printf '%s\n' "watchdog_trip=1" >>"$IMAGE_LOG"
  else
    printf '%s\n' "watchdog_trip=0" >>"$IMAGE_LOG"
  fi
  cleanup
  if [[ "$bench_status" != 0 ]]; then
    return "$bench_status"
  fi
}

select_model
for repeat in 1 2; do
  run_repeat "$repeat"
done
aggregate_inputs=("$RESULT_DIR/$SLUG.repeat1.json" "$RESULT_DIR/$SLUG.repeat2.json")
set +e
python3 scripts/aggregate-repeats.py --slug "$SLUG" --output "$RESULT_DIR/$SLUG.json" --samples-output "$RESULT_DIR/$SLUG.samples.json" "${aggregate_inputs[@]}"
aggregate_status=$?
set -e
if [[ "$aggregate_status" == 3 ]]; then
  run_repeat 3
  aggregate_inputs+=("$RESULT_DIR/$SLUG.repeat3.json")
  python3 scripts/aggregate-repeats.py --slug "$SLUG" --output "$RESULT_DIR/$SLUG.json" --samples-output "$RESULT_DIR/$SLUG.samples.json" "${aggregate_inputs[@]}"
elif [[ "$aggregate_status" != 0 ]]; then
  exit "$aggregate_status"
date '+%Y-%m-%d %H:%M %z' > "$LOG_DIR/completed-$SLUG.txt"
fi

