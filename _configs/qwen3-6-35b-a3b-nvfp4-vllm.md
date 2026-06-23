---
title: Qwen3.6-35B-A3B · vLLM · NVFP4
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: NVIDIA's OFFICIAL NVFP4 of the Qwen3.6-35B-A3B sparse-MoE (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) — preferred over the unsloth quant per policy (use the nvidia image when one exists). Published near-baseline accuracy (MMLU-Pro 85.0 vs 85.6 BF16). Base (non-speculative); the repo's MTP module is exercised in the -mtp sibling. Compare NVFP4 vs the official FP8 sweep.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). NVIDIA's official ModelOpt NVFP4 — nvidia documents a DGX Spark recipe (vLLM).
  # Base form; benchmarked at conc-32 / util 0.85 (the nvidia DGX example uses util 0.4 / max-num-seqs 4 —
  # tuned for headroom, not throughput; we sweep concurrency instead). modelopt quant + qwen3 reasoning parser.
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 32 (wrapper defaults)
---

**Queued — Qwen3.6-35B-A3B NVFP4 base on vLLM, using NVIDIA's official ModelOpt quant.** First NVFP4
datapoint for the sparse-MoE 3.6 (only the official FP8 sweep existed); direct NVFP4-vs-FP8 compare
against [Qwen3.6-35B-A3B · vLLM · FP8].

- **Repo — NVIDIA official (per policy).** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4),
  NVIDIA ModelOpt v0.44.0, **published near-baseline accuracy** (MMLU-Pro 85.0 vs 85.6 BF16, GPQA-Diamond
  84.8 vs 84.9, AIME'25 88.8 vs 89.2). Chosen over unsloth because an official `nvidia/` NVFP4 exists —
  and NVIDIA documents a **DGX Spark recipe** for it (hence the `Spark recipe` tag).
- **NVIDIA DGX Spark recipe flags (from the card)** for reference: `--quantization modelopt
  --reasoning-parser qwen3 --trust-remote-code --kv-cache-dtype fp8 --attention-backend flashinfer
  --moe-backend marlin --load-format fastsafetensors --enable-chunked-prefill --enable-prefix-caching
  --async-scheduling`. The card's example pins util 0.4 / max-num-seqs 4 (headroom-tuned); we run the
  wrapper's util 0.85 + concurrency sweep instead.
- **Spec-decode:** repo ships MTP → `-mtp` sibling (needs the marlin/triton MoE-backend pair, see there).
- **Grid mirrors the 27B NVFP4 set:** `{vLLM, SGLang} × {base, MTP}`.
