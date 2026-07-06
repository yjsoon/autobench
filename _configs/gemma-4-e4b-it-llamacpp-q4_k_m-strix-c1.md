---
title: Gemma 4 E4B · llama.cpp · Q4_K_M · Strix Halo · conc 1
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: lmstudio-community Q4_K_M — the GGUF already on disk from LM Studio; same quant family as the Spark baseline run (which used unsloth/ggml-org quants).
source_repo: lmstudio-community/gemma-4-E4B-it-GGUF
download_url: https://huggingface.co/lmstudio-community/gemma-4-E4B-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B, conc-1, strix-halo]
status: done
prefill_toks: 71.57
decode_toks: 54.97
mem_gb: 7.49
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — peak 5.77 GiB, delta 4.39 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool)
vram_peak_gb: 5.77
vram_delta_gb: 4.39
measured_on: 2026-07-04
completed_at: 2026-07-04 07:16 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh \
    lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 195/1000 prompts (hit 900 s cap), 0 errors.
---

**First Strix Halo serving run of Gemma 4 E4B** — single stream, full 65536 context in one slot.
Part of the overnight 2026-07-04 llama.cpp Vulkan sweep (`results/overnight-20260704-070027/`).

- **Result (conc 1):** prefill 71.57 / decode **54.97** tok/s; 195/1000 prompts (hit the 900 s cap),
  **0 errors**.
- **Memory:** both counters moved — MemAvailable delta 7.49 GB (headline), VRAM pool delta 4.39 GiB
  (peak 5.77 GiB) for the 4.95 GiB GGUF + KV.
- **Synthetic ceiling** (same night, llama-bench pp512/tg128): prefill 2099.6 ± 29.9 / decode
  61.2 ± 0.4 tok/s — serving decode at c1 reaches ~90% of the synthetic tg128 rate.
- Sweep siblings: [`-c8`](gemma-4-e4b-it-llamacpp-q4_k_m-strix-c8) ·
  [`-c32`](gemma-4-e4b-it-llamacpp-q4_k_m-strix-c32). Spark baseline (same engine/quant, conc 32):
  [`gemma-4-e4b-it-llamacpp-q4_k_m`](gemma-4-e4b-it-llamacpp-q4_k_m).
