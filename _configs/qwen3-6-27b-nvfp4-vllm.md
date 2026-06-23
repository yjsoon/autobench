---
title: Qwen3.6-27B · vLLM · NVFP4
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4), calibrated on HF UltraChat @16K. NVFP4 is a genuine GB10 fast-path (see INCOMPATIBILITIES.md) — benchmark it against the official FP8 run on the same model. Base (non-speculative) config; the repo's MTP module is exercised in the -mtp sibling.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). vLLM is unsloth's recommended path for this NVFP4 checkpoint.
  # Per the new image policy, try nightly-aarch64 first (fall back to cu130-nightly only if it regresses).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:nightly-aarch64 unsloth/Qwen3.6-27B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --trust-remote-code --dtype bfloat16
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model unsloth/Qwen3.6-27B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Queued — Qwen3.6-27B NVFP4 base run on vLLM.** Unsloth's NVFP4 (W4A4) quant, served text-only on
vLLM (the card's recommended path for this checkpoint). Direct NVFP4-vs-FP8 comparison against the
official [Qwen3.6-27B · vLLM · FP8] run at the same model/context/concurrency.

- **Model:** `unsloth/Qwen3.6-27B-NVFP4`, NVFP4 calibrated on HF UltraChat @16K. Native context 262K
  (YaRN-extendable to ~1M); benchmarked at **65536** to match the FP8 sweep and fit 121 GB unified.
- **Spec-decode:** the repo ships an MTP module → exercised separately in the `-mtp` sibling, not here.
- **Pair:** base (this) + `qwen3-6-27b-nvfp4-vllm-mtp` (with MTP), mirroring the FP8 base+MTP pages.
- **Repo choice — unsloth (no NVIDIA option).** Policy is to prefer an official `nvidia/` NVFP4 when one
  exists, but **NVIDIA publishes no NVFP4 for the 27B** (only for the 35B-A3B sibling); the 27B has only
  community NVFP4 quants (unsloth, mmangkad, sakamakismile, …). So this uses **unsloth** (a trusted,
  well-known quantizer per the repo policy).
