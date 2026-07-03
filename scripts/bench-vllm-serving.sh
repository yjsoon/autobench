#!/usr/bin/env bash
# Serving benchmark for one HF model via vLLM, driven by scripts/bench-serving.py
# against the ShareGPT workload.
#
# Strix Halo port — EXPERIMENTAL: gfx1151 is not in upstream vLLM's supported list.
# Default image is rocm/vllm:latest; expect to need HSA_OVERRIDE_GFX_VERSION=11.5.1
# (exported before running, it is passed through) or a custom gfx1151 build
# (see https://blog.epheo.eu/notes/strix-halo/index.html). Get llama.cpp and SGLang
# working first; treat failures here as expected until the stack matures.
#
# vLLM exposes an OpenAI-compatible API on :8000, so the same client works as for
# llama.cpp / SGLang. Headline memory = system MemAvailable delta from idle (10s sampling).
# NOTE: like SGLang, vLLM pre-reserves a static KV fraction (--gpu-memory-utilization,
# default here 0.85 of unified mem) — so mem_gb reflects that reservation, not the model
# footprint. See the config Notes for the per-model resident breakdown from the vLLM logs.
#
# Usage: bench-vllm-serving.sh <hf-model-path> [ctx] [conc] [num_prompts] [max_seconds] [max_tokens] [extra vllm args...]
#   e.g. bench-vllm-serving.sh nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4 65536 32 1000 900 256 \
#          --trust-remote-code --reasoning-parser nemotron_v3
set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env 2>/dev/null || true; set +a

DATASET="$(pwd)/benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json"
BENCH="$(pwd)/scripts/bench-serving.py"
# Default to AMD's ROCm vLLM image; override with VLLM_IMAGE (e.g. a custom gfx1151 build).
IMAGE="${VLLM_IMAGE:-rocm/vllm:latest}"
PORT=8000

MODEL="${1:?need an HF model path, e.g. nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4}"
CTX="${2:-65536}"; CONC="${3:-32}"; NUMP="${4:-1000}"; MAXS="${5:-900}"; MAXTOK="${6:-256}"
shift 6 2>/dev/null || shift $#
EXTRA=("$@")   # any extra `vllm serve` flags (parsers, --trust-remote-code, etc.)
NAME="vllm-$(echo "$MODEL" | tr -c 'A-Za-z0-9' '-')"
gib_kb() { awk '{printf "%.2f", $1/1024/1024}'; }

base_avail=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
docker rm -f "$NAME" >/dev/null 2>&1 || true

# NB: this image's ENTRYPOINT is ["vllm","serve"], so the command is just <MODEL> <flags>
# (do NOT prepend `vllm serve` — that doubles it and argparse rejects it).
echo "==> launch vLLM $MODEL (ctx=$CTX, max-num-seqs=$CONC) extra=[${EXTRA[*]:-}]"
# gpt-oss / harmony models need the o200k_base tiktoken vocab. The rust openai_harmony
# lib fetches it from $TIKTOKEN_ENCODINGS_BASE (default = openaipublic blob). Pre-seed it
# from a local mount so the load needs NO second egress (the sandbox caps outbound after
# the first connection, which the HF model download already used). Plain path, not file://.
# Harmony vocab dir mounted at /vocab. Default is the historical path; override with VOCAB_DIR=
# (e.g. VOCAB_DIR=$HOME/tiktoken_encodings) when that path is missing/root-owned — the dir just
# needs to contain o200k_base.tiktoken (sha256 446a9538…). See notes/INCOMPATIBILITIES.md.
VOCAB_DIR="${VOCAB_DIR:-$HOME/models/tiktoken_cache}"
VOCAB_ARGS=()
[ -d "$VOCAB_DIR" ] && VOCAB_ARGS=(-v "$VOCAB_DIR:/vocab:ro" --env "TIKTOKEN_ENCODINGS_BASE=/vocab")
docker run -d --name "$NAME" --device /dev/kfd --device /dev/dri \
  --ipc=host --security-opt seccomp=unconfined -p "$PORT:$PORT" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  "${VOCAB_ARGS[@]}" \
  --env "HF_TOKEN=${HF_TOKEN:-}" \
  --env "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-}" \
  "$IMAGE" "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --max-model-len "$CTX" --gpu-memory-utilization 0.85 --max-num-seqs "$CONC" \
  "${EXTRA[@]}" >/dev/null

cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "==> waiting for /health (model download+load can be many minutes) ..."
for i in $(seq 1 1800); do
  if curl -fsS "http://localhost:$PORT/health" >/dev/null 2>&1; then echo "    ready after ${i}s"; break; fi
  if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" != "true" ]; then
    echo "!! vLLM container exited during load:"; docker logs "$NAME" 2>&1 | tail -50; exit 1
  fi
  sleep 1
  [ "$i" = 1800 ] && { echo "!! health timeout"; docker logs "$NAME" 2>&1 | tail -50; exit 1; }
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

# Speculative-decoding acceptance metrics — pulled from the engine log BEFORE teardown (the trap
# removes the container). vLLM logs e.g. "Spec decode: ... mean acceptance length: X.XX" / per-pos
# "acceptance rate". Record on the config page for any --speculative-config run; empty for base runs.
if [ -n "${EXTRA[*]:-}" ] && printf '%s\n' "${EXTRA[@]}" | grep -q "speculative"; then
  echo "==> SPEC-METRICS (acceptance):"
  docker logs "$NAME" 2>&1 | grep -iE "accept|spec.?decode|num_spec|draft" | tail -20 || true
fi
