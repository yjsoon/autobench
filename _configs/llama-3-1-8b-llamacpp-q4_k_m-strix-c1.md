---
title: Llama 3.1 8B · llama.cpp · Q4_K_M · Strix Halo · conc 1
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
concurrency: 1
tags: [llama-3-1-8b, Meta, Llama, Q4_K_M, 5-15B, conc-1, strix-halo]
status: done
prefill_toks: 53.08
decode_toks: 40.5
mem_gb: 12.47
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 36.61 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 36.61
vram_delta_gb: 12.47
measured_on: 2026-07-14
completed_at: 2026-07-13 23:47:02 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 166/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The classic 8B baseline — the strongest concurrency scaler here (956/1000 completed at c32).**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 53.08 / decode **40.5** tok/s;
  166/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 1270.19 / decode **45.81** tok/s.
- **Spark same-engine baseline:** [`llama-3-1-8b-llamacpp-q4_k_m`](llama-3-1-8b-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 365.22).
- **Memory:** VRAM pool delta **12.47 GiB** (peak 36.61 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](llama-3-1-8b-llamacpp-q4_k_m-strix-c8) · [`-c32`](llama-3-1-8b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/llama-3p1-8b-q4km/`.
