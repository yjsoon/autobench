---
title: Llama 3.1 8B · llama.cpp · Q4_K_M · Strix Halo · conc 8
model: meta-llama/Llama-3.1-8B-Instruct
company: Meta
family: Llama
params: 8B
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from bartowski — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
download_url: https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 8
tags: [llama-3-1-8b, Meta, Llama, Q4_K_M, 5-15B, conc-8, strix-halo]
status: done
prefill_toks: 169.15
decode_toks: 150.5
mem_gb: 12.41
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 36.56 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 36.56
vram_delta_gb: 12.41
measured_on: 2026-07-14
completed_at: 2026-07-14 00:02:15 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 8 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf 65536 8 1000 900 256 99
  # 629/1000 prompts (hit the 900 s cap), 1 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The throughput case — aggregate decode with continuous batching.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 8):** prefill 169.15 / decode **150.5** tok/s aggregate;
  629/1000 prompts (hit the 900 s cap), **1 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 1270.19 / decode **45.81** tok/s.
- **Spark same-engine baseline:** [`llama-3-1-8b-llamacpp-q4_k_m`](llama-3-1-8b-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 365.22).
- **Memory:** VRAM pool delta **12.41 GiB** (peak 36.56 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](llama-3-1-8b-llamacpp-q4_k_m-strix-c1) · [`-c32`](llama-3-1-8b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/llama-3p1-8b-q4km/`.
