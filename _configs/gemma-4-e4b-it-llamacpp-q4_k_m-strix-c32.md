---
title: Gemma 4 E4B · llama.cpp · Q4_K_M · Strix Halo · conc 32
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
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B, conc-32, strix-halo]
status: done
prefill_toks: 237.33
decode_toks: 314.58
mem_gb: 9.45
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — peak 6.91 GiB, delta 5.18 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool)
vram_peak_gb: 6.91
vram_delta_gb: 5.18
measured_on: 2026-07-04
completed_at: 2026-07-04 07:44 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  # NOTE: llama.cpp splits -c across slots → only 2048 tokens per request at conc 32.
  scripts/bench-llamacpp-serving.sh \
    lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 980/1000 prompts in 780.8 s (finished under the cap), 20 errors (slot-split).
---

**The direct Spark head-to-head** — same engine, same quant, same conc 32, same 980/1000 + 20-error
finish as the Spark run. Part of the overnight 2026-07-04 llama.cpp Vulkan sweep
(`results/overnight-20260704-070027/`).

- **Result (conc 32):** prefill 237.33 / decode **314.58** tok/s aggregate; 980/1000 prompts in
  **780.8 s** (finished under the 900 s cap), 20 errors (slot-split — 2048 tokens/request at conc 32).
- **vs Spark** ([`gemma-4-e4b-it-llamacpp-q4_k_m`](gemma-4-e4b-it-llamacpp-q4_k_m), CUDA, conc 32:
  328.95 prefill / 435.01 decode in 563 s): Strix Halo Vulkan delivers **~72% of Spark decode** on
  the identical workload — the cleanest cross-machine datapoint so far.
- **Memory:** MemAvailable delta 9.45 GB (headline), VRAM pool delta 5.18 GiB (peak 6.91 GiB).
- Sweep siblings: [`-c1`](gemma-4-e4b-it-llamacpp-q4_k_m-strix-c1) ·
  [`-c8`](gemma-4-e4b-it-llamacpp-q4_k_m-strix-c8).
