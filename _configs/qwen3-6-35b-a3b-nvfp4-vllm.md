---
title: Qwen3.6-35B-A3B · vLLM · NVFP4
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: NVIDIA's OFFICIAL NVFP4 of the Qwen3.6-35B-A3B sparse-MoE (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) — preferred over the unsloth quant per policy (use the nvidia image when one exists). Mixed-precision NVFP4 (FP8 backbone + NVFP4/W4A16 experts). Published near-baseline accuracy (MMLU-Pro 85.0 vs 85.6 BF16). Base run beat the official FP8 sweep by ~50% decode.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-32]
status: done
prefill_toks: 443.78
decode_toks: 430.76
mem_gb: 103.84
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85)
measured_on: 2026-06-23
completed_at: 2026-06-23 11:13 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # NVIDIA official ModelOpt NVFP4 MoE (Resolved arch: Qwen3_5MoeForConditionalGeneration;
  # quant_algo NVFP4 + FP8 backbone + W4A16_NVFP4 experts) on vLLM nightly-aarch64. Base, conc-32.
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3
  # 1000/1000 prompts, 0 errors, 593.9 s (no time cap). Aggregate prefill 443.8 / decode 430.8 tok/s.
  # NOTE: per-stream TTFT (~18 s) / TPOT (0.0) are distorted by the qwen3 reasoning stream + conc-32
  # queueing — the aggregate decode rate is the valid metric here.
---

**NVFP4 MoE flies — decode 430.8 tok/s, +50% over the official FP8 base.** NVIDIA's ModelOpt NVFP4 of the
Qwen3.6-35B-A3B sparse-MoE on vLLM, the first NVFP4 datapoint for this model (only the FP8 sweep existed).

- **Result (conc 32):** prefill **443.8** / decode **430.76** tok/s aggregate; **1000/1000, 0 errors** in
  593.9 s (no time cap — the MoE clears the full count comfortably). Peak mem **103.8 GB** (slightly under
  the FP8 base's 107.9 — NVFP4 experts shrink the weights; KV reservation still dominates the headline).
- **NVFP4 vs FP8:** decode **430.8 vs 286.0** (the [FP8 base]) = **+50%** — the biggest NVFP4 win in the
  sweep, as expected for a 3B-active MoE where the 4-bit expert matmuls dominate decode.
- **Repo — NVIDIA official (per policy):** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4),
  ModelOpt v0.44.0, mixed-precision (FP8 backbone + NVFP4/W4A16 experts). NVIDIA documents a DGX Spark
  recipe for it → `Spark recipe` tag. Published near-baseline accuracy (MMLU-Pro 85.0 vs 85.6 BF16).
- **Measurement caveat:** `--reasoning-parser qwen3` makes the model stream a thinking trace; combined with
  conc-32 queueing this distorts per-stream TTFT (~18 s) and zeros the TPOT median. Aggregate tok/s is
  unaffected and is the reported figure.
- **Pair:** base (this) + `qwen3-6-35b-a3b-nvfp4-vllm-mtp` (MTP). SGLang siblings blocked (spark image
  arch wall — see those pages / a newer SGLang nightly under evaluation).
