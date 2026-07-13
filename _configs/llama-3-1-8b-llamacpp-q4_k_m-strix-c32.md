---
title: Llama 3.1 8B · llama.cpp · Q4_K_M · Strix Halo · conc 32
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
concurrency: 32
tags: [llama-3-1-8b, Meta, Llama, Q4_K_M, 5-15B, conc-32, strix-halo]
status: done
prefill_toks: 201.14
decode_toks: 221.17
mem_gb: 12.41
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 36.55 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 36.55
vram_delta_gb: 12.41
measured_on: 2026-07-14
completed_at: 2026-07-14 00:17:58 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 956/1000 prompts (hit the 900 s cap), 15 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**Peak concurrency — where the slot-split (2048 tokens/request) starts costing errors.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 32):** prefill 201.14 / decode **221.17** tok/s aggregate;
  956/1000 prompts (hit the 900 s cap), **15 errors** (slot-split — 2048 tokens/request).
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 1270.19 / decode **45.81** tok/s.
- **vs Spark** ([`llama-3-1-8b-llamacpp-q4_k_m`](llama-3-1-8b-llamacpp-q4_k_m), llama.cpp CUDA Q4_K_M c32 decode 365.22): Strix Halo Vulkan reaches **61%** on the identical engine/quant/workload.
- **Memory:** VRAM pool delta **12.41 GiB** (peak 36.55 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](llama-3-1-8b-llamacpp-q4_k_m-strix-c1) · [`-c8`](llama-3-1-8b-llamacpp-q4_k_m-strix-c8). Evidence: `results/batch-20260713-225032/llama-3p1-8b-q4km/`.
