---
title: DeepSeek-R1-Distill-Qwen-14B · llama.cpp · Q4_K_M · Strix Halo · conc 1
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
company: DeepSeek
family: DeepSeek-R1-Distill
params: 14B
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from bartowski — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
download_url: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [deepseek-r1-distill-qwen-14b, DeepSeek, DeepSeek-R1-Distill, Q4_K_M, 5-15B, conc-1, strix-halo]
status: done
prefill_toks: 36.26
decode_toks: 21.83
mem_gb: 20.12
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 44.26 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 44.26
vram_delta_gb: 20.12
measured_on: 2026-07-14
completed_at: 2026-07-14 01:21:12 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 80/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The middle rung of the R1-distill dense ladder (8B→14B→32B) — clean scaling.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 36.26 / decode **21.83** tok/s;
  80/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 743.67 / decode **24.89** tok/s.
- **Spark same-engine baseline:** [`deepseek-r1-distill-qwen-14b-llamacpp-q4_k_m`](deepseek-r1-distill-qwen-14b-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 243.75).
- **Memory:** VRAM pool delta **20.12 GiB** (peak 44.26 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](deepseek-r1-distill-qwen-14b-llamacpp-q4_k_m-strix-c8) · [`-c32`](deepseek-r1-distill-qwen-14b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/deepseek-r1-distill-qwen-14b-q4km/`.
