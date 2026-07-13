---
title: Gemma 4 12B · llama.cpp · Q4_K_M · Strix Halo · conc 1
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
concurrency: 1
tags: [gemma-4-12b, Google, Gemma, Q4_K_M, 5-15B, conc-1, strix-halo]
status: done
prefill_toks: 38.97
decode_toks: 23.81
mem_gb: 8.27
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 32.27 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 32.27
vram_delta_gb: 8.27
measured_on: 2026-07-14
completed_at: 2026-07-14 02:08:24 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/gemma-4-12b-it-GGUF/gemma-4-12b-it-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 85/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The cleanest cross-machine control: same engine, same Q4_K_M, same model as a Spark llama.cpp run.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 38.97 / decode **23.81** tok/s;
  85/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 803.85 / decode **28.45** tok/s.
- **Spark same-engine baseline:** [`gemma-4-12b-it-llamacpp-q4_k_m`](gemma-4-12b-it-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 195.25).
- **Memory:** VRAM pool delta **8.27 GiB** (peak 32.27 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](gemma-4-12b-llamacpp-q4_k_m-strix-c8) · [`-c32`](gemma-4-12b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/gemma-4-12b-q4km/`.
