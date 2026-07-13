---
title: Gemma 4 12B · llama.cpp · Q4_K_M · Strix Halo · conc 32
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/gemma-4-12b-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-12b-it-GGUF
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, Q4_K_M, 5-15B, conc-32, strix-halo]
status: done
prefill_toks: 91.82
decode_toks: 119.68
mem_gb: 22.74
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 46.74 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 46.74
vram_delta_gb: 22.74
measured_on: 2026-07-14
completed_at: 2026-07-14 02:39:54 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 452/1000 prompts (hit the 900 s cap), 10 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**Peak concurrency — where the slot-split (2048 tokens/request) starts costing errors.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 32):** prefill 91.82 / decode **119.68** tok/s aggregate;
  452/1000 prompts (hit the 900 s cap), **10 errors** (slot-split — 2048 tokens/request).
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 803.85 / decode **28.45** tok/s.
- **vs Spark** ([`gemma-4-12b-it-llamacpp-q4_k_m`](gemma-4-12b-it-llamacpp-q4_k_m), llama.cpp CUDA Q4_K_M c32 decode 195.25): Strix Halo Vulkan reaches **61%** on the identical engine/quant/workload.
- **Memory:** VRAM pool delta **22.74 GiB** (peak 46.74 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](gemma-4-12b-llamacpp-q4_k_m-strix-c1) · [`-c8`](gemma-4-12b-llamacpp-q4_k_m-strix-c8). Evidence: `results/batch-20260713-225032/gemma-4-12b-q4km/`.
