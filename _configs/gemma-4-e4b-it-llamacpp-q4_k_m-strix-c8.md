---
title: Gemma 4 E4B · llama.cpp · Q4_K_M · Strix Halo · conc 8
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
concurrency: 8
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B, conc-8, strix-halo]
status: done
prefill_toks: 197.13
decode_toks: 204.44
mem_gb: 9.35
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — peak 5.98 GiB, delta 4.25 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool)
vram_peak_gb: 5.98
vram_delta_gb: 4.25
measured_on: 2026-07-04
completed_at: 2026-07-04 07:31 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 8 -cb + bench-serving.py (ShareGPT V3).
  # NOTE: llama.cpp splits -c across slots → 8192 tokens per request at conc 8.
  scripts/bench-llamacpp-serving.sh \
    lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf 65536 8 1000 900 256 99
  # 737/1000 prompts (hit 900 s cap), 1 error (slot-split).
---

**Gemma 4 E4B at conc 8 on Strix Halo** — decode aggregate ~3.7× the single-stream rate.
Part of the overnight 2026-07-04 llama.cpp Vulkan sweep (`results/overnight-20260704-070027/`).

- **Result (conc 8):** prefill 197.13 / decode **204.44** tok/s aggregate; 737/1000 prompts
  (hit the 900 s cap), **1 error**.
- **Memory:** MemAvailable delta 9.35 GB (headline), VRAM pool delta 4.25 GiB (peak 5.98 GiB) —
  batching adds host-side overhead but barely moves the VRAM footprint.
- **Slot-split caveat:** 65536 ctx across 8 slots = 8192 tokens per request — fine for ShareGPT,
  too small for long-context work.
- Sweep siblings: [`-c1`](gemma-4-e4b-it-llamacpp-q4_k_m-strix-c1) ·
  [`-c32`](gemma-4-e4b-it-llamacpp-q4_k_m-strix-c32). Spark baseline (same engine/quant, conc 32):
  [`gemma-4-e4b-it-llamacpp-q4_k_m`](gemma-4-e4b-it-llamacpp-q4_k_m).
