#!/usr/bin/env bash
# DDTree / DFlash single-stream research-harness benchmark (batch-1, bf16 target).
#
# The tree-decoding method (Ringel & Romano 2026, github.com/liranringel/ddtree) is NOT in any
# serving engine (vLLM/SGLang), so this runs the paper's PyTorch harness inside a trusted torch
# container (SGLang image = torch cu130 + flash_attn + datasets, all built for GB10 sm_121a).
#
# In ONE pass on the SAME prompts the harness measures three methods:
#   baseline    = block_size 1  = plain autoregressive target (the true single-stream base)
#   dflash      = single-line block-diffusion draft (num spec tokens = block_size)
#   ddtree_tb*  = tree draft over the block-diffusion distribution at each --tree-budget
# -> per-method accept-len + single-stream decode tok/s, a clean controlled DFlash-vs-DDTree compare.
#
# METHODOLOGY LIMITS (record on the config page):
#   * batch-1 only. This harness generates one request at a time (torchrun just data-parallel-splits
#     the dataset); it does NOT batch concurrent requests, so conc-8/32 are not measurable here.
#   * bf16 target, NOT NVFP4. AutoModelForCausalLM loads the unquantized bf16 checkpoint. So absolute
#     tok/s is NOT comparable to the NVFP4 serving rows; the transferable numbers are the
#     DDTree-vs-DFlash ACCEPT-LEN ratio and the single-stream speedup-vs-autoregressive-baseline.
#
# Usage: bench-ddtree.sh <hf-target> <hf-dflash-drafter> [dataset] [samples] [max_new] [budgets] [temp] [label]
#   scripts/bench-ddtree.sh Qwen/Qwen3.6-35B-A3B z-lab/Qwen3.6-35B-A3B-DFlash mt-bench 20 512 64,256 0.0 q35b-mtbench
set -uo pipefail
TARGET=${1:?hf target repo}
DRAFT=${2:?hf DFlash drafter repo}
DATASET=${3:-mt-bench}
SAMPLES=${4:-20}
MAXNEW=${5:-512}
BUDGETS=${6:-64,256}
TEMP=${7:-0.0}
LABEL=${8:-ddtree}

IMG=${DDTREE_IMAGE:-lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed}
REPO=${DDTREE_REPO:-$HOME/ddtree}
RUNS=${RUNS_DIR:-$HOME/ddtree/runs}
HFCACHE=${DDTREE_HF_CACHE:-$HOME/ddtree/hf_cache}   # persistent across --rm (container writes as root)
SAVE="/work/runs/${LABEL}.pt"
CNAME=bench-ddtree
mkdir -p "$RUNS" "$HFCACHE"

# load HF_TOKEN (public repos, but harmless) if present
[ -f "$(dirname "$0")/../.env" ] && { set -a; . "$(dirname "$0")/../.env"; set +a; }

echo "==> DDTree harness: target=$TARGET draft=$DRAFT dataset=$DATASET samples=$SAMPLES max_new=$MAXNEW budgets=$BUDGETS temp=$TEMP"
echo "==> image=$IMG  (batch-1, bf16 target)"

# idle MemAvailable baseline (kB)
base_kb=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
peak_file=$(mktemp)
( min_kb=$base_kb
  for _ in $(seq 1 720); do
    cur=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$cur" -lt "$min_kb" ] && min_kb=$cur
    echo "$min_kb" > "$peak_file"
    sleep 10
  done ) &
SAMPLER=$!

docker rm -f "$CNAME" >/dev/null 2>&1 || true
docker run --rm --name "$CNAME" --gpus all --ipc=host \
  -v "$REPO":/work -w /work \
  -v "$HFCACHE":/hf \
  -e HF_HOME=/hf -e HF_TOKEN="${HF_TOKEN:-}" -e HF_HUB_ENABLE_HF_TRANSFER=0 \
  --entrypoint python3 "$IMG" \
  benchmark.py \
    --model-name-or-path "$TARGET" \
    --draft-name-or-path "$DRAFT" \
    --dataset "$DATASET" \
    --max-samples "$SAMPLES" \
    --max-new-tokens "$MAXNEW" \
    --tree-budget "$BUDGETS" \
    --temperature "$TEMP" \
    --disable-cpp-compact-cache \
    --save-path "$SAVE"
rc=$?

kill "$SAMPLER" 2>/dev/null || true
min_kb=$(cat "$peak_file" 2>/dev/null || echo "$base_kb")
rm -f "$peak_file"
peak_gb=$(awk -v b="$base_kb" -v m="$min_kb" 'BEGIN{printf "%.2f", (b-m)/1024/1024}')

echo "==> exit rc=$rc  peak mem delta=${peak_gb} GB (system MemAvailable, 10s sampling)"
if [ "$rc" -eq 0 ]; then
  echo "==> metrics:"
  docker run --rm -v "$REPO":/work -w /work --entrypoint python3 "$IMG" ddtree_metrics.py "$SAVE"
  echo "MEM mem_gb=${peak_gb} mem_source=\"system MemAvailable delta (10s sampling)\""
fi
