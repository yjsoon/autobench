---
title: Gemma 4 31B · llama.cpp · Q4_K_M · Strix Halo · conc 32
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 31B (dense)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/gemma-4-31B-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-31b, Google, Gemma, Q4_K_M, 16-40B, conc-32, strix-halo]
status: done
prefill_toks: 40.21
decode_toks: 49.29
mem_gb: 54.86
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 83.73 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 83.73
vram_delta_gb: 54.86
measured_on: 2026-07-14
completed_at: 2026-07-14 05:02:42 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 194/1000 prompts (hit the 900 s cap), 10 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**Peak concurrency — where the slot-split (2048 tokens/request) starts costing errors.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 32):** prefill 40.21 / decode **49.29** tok/s aggregate;
  194/1000 prompts (hit the 900 s cap), **10 errors** (slot-split — 2048 tokens/request).
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 300.11 / decode **11.68** tok/s.
- **vs Spark** ([`gemma-4-31b-it-llamacpp-q4_k_m`](gemma-4-31b-it-llamacpp-q4_k_m), llama.cpp CUDA Q4_K_M c32 decode 78.45): Strix Halo Vulkan reaches **63%** on the identical engine/quant/workload.
- **Memory:** VRAM pool delta **54.86 GiB** (peak 83.73 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](gemma-4-31b-llamacpp-q4_k_m-strix-c1) · [`-c8`](gemma-4-31b-llamacpp-q4_k_m-strix-c8). Evidence: `results/batch-20260713-225032/gemma-4-31b-q4km/`.
