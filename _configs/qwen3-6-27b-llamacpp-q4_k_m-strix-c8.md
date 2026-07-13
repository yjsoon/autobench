---
title: Qwen3.6 27B · llama.cpp · Q4_K_M · Strix Halo · conc 8
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/Qwen3.6-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 8
tags: [qwen3.6-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-8, strix-halo]
status: done
prefill_toks: 40.24
decode_toks: 35.75
mem_gb: 20.35
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 44.35 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 44.35
vram_delta_gb: 20.35
measured_on: 2026-07-14
completed_at: 2026-07-14 03:11:37 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 8 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf 65536 8 1000 900 256 99
  # 128/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The throughput case — aggregate decode with continuous batching.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 8):** prefill 40.24 / decode **35.75** tok/s aggregate;
  128/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 361.82 / decode **12.77** tok/s.
- Spark ran Qwen3.6-27B only on vLLM/SGLang (NVFP4/FP8), not llama.cpp — no same-engine baseline; see the [model tag](../tags/model/) for the Spark configs.
- **Memory:** VRAM pool delta **20.35 GiB** (peak 44.35 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](qwen3-6-27b-llamacpp-q4_k_m-strix-c1) · [`-c32`](qwen3-6-27b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/qwen3p6-27b-q4km/`.
