---
title: DeepSeek-R1-Distill-Qwen-32B · llama.cpp · Q4_K_M · Strix Halo · conc 32
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
company: DeepSeek
family: DeepSeek-R1-Distill
params: 32B
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from bartowski — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
download_url: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 32
tags: [deepseek-r1-distill-qwen-32b, DeepSeek, DeepSeek-R1-Distill, Q4_K_M, 16-40B, conc-32, strix-halo]
status: done
prefill_toks: 60.98
decode_toks: 63.74
mem_gb: 22.07
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 58.26 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 58.26
vram_delta_gb: 22.07
measured_on: 2026-07-14
completed_at: 2026-07-14 05:52:47 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 268/1000 prompts (hit the 900 s cap), 8 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**Peak concurrency — where the slot-split (2048 tokens/request) starts costing errors.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 32):** prefill 60.98 / decode **63.74** tok/s aggregate;
  268/1000 prompts (hit the 900 s cap), **8 errors** (slot-split — 2048 tokens/request).
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 321.57 / decode **11.4** tok/s.
- **vs Spark** ([`deepseek-r1-distill-qwen-32b-llamacpp-q4_k_m`](deepseek-r1-distill-qwen-32b-llamacpp-q4_k_m), llama.cpp CUDA Q4_K_M c32 decode 117.59): Strix Halo Vulkan reaches **54%** on the identical engine/quant/workload.
- **Memory:** VRAM pool delta **22.07 GiB** (peak 58.26 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](deepseek-r1-distill-qwen-32b-llamacpp-q4_k_m-strix-c1) · [`-c8`](deepseek-r1-distill-qwen-32b-llamacpp-q4_k_m-strix-c8). Evidence: `results/batch-20260713-225032/deepseek-r1-distill-qwen-32b-q4km/`.
