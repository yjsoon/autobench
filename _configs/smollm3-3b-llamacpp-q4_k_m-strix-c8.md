---
title: SmolLM3 3B · llama.cpp · Q4_K_M · Strix Halo · conc 8
model: HuggingFaceTB/SmolLM3-3B
company: HuggingFaceTB
family: SmolLM3
params: 3B
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from ggml-org — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: ggml-org/SmolLM3-3B-GGUF
download_url: https://huggingface.co/ggml-org/SmolLM3-3B-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 8
tags: [smollm3-3b, HuggingFaceTB, SmolLM3, Q4_K_M, ≤4B, conc-8, strix-halo]
status: done
prefill_toks: 479.06
decode_toks: 269.54
mem_gb: 6.36
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 30.51 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 30.51
vram_delta_gb: 6.36
measured_on: 2026-07-14
completed_at: 2026-07-13 23:21:01 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 8 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh ggml-org/SmolLM3-3B-GGUF/SmolLM3-Q4_K_M.gguf 65536 8 1000 900 256 99
  # 966/1000 prompts (hit the 900 s cap), 2 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The throughput case — aggregate decode with continuous batching.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 8):** prefill 479.06 / decode **269.54** tok/s aggregate;
  966/1000 prompts (hit the 900 s cap), **2 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 3088.07 / decode **104.59** tok/s.
- **Spark same-engine baseline:** [`smollm3-3b-llamacpp-q4_k_m`](smollm3-3b-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 653.55).
- **Memory:** VRAM pool delta **6.36 GiB** (peak 30.51 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](smollm3-3b-llamacpp-q4_k_m-strix-c1) · [`-c32`](smollm3-3b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/smollm3-3b-q4km/`.
