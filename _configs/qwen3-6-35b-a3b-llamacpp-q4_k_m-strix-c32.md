---
title: Qwen3.6-35B-A3B · llama.cpp · Q4_K_M · Strix Halo · conc 32
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: lmstudio-community Q4_K_M (19.70 GiB) — the GGUF already on disk from LM Studio; nearest GGUF equivalent to the Spark's NVFP4 (Blackwell-only format) for cross-machine comparison.
source_repo: lmstudio-community/Qwen3.6-35B-A3B-GGUF
download_url: https://huggingface.co/lmstudio-community/Qwen3.6-35B-A3B-GGUF
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-32, strix-halo]
status: done
prefill_toks: 79.90
decode_toks: 96.86
mem_gb: 11.15
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — peak 24.59 GiB, delta 22.86 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool) — the VRAM counter is the one that moved for the model itself
vram_peak_gb: 24.59
vram_delta_gb: 22.86
measured_on: 2026-07-04
completed_at: 2026-07-04 08:31 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  # NOTE: llama.cpp splits -c across slots → only 2048 tokens per request at conc 32.
  scripts/bench-llamacpp-serving.sh \
    lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 355/1000 prompts (hit 900 s cap), 10 errors (slot-split).
---

**Past the saturation point** — conc 32 buys nothing over conc 8 for this MoE on Strix Halo Vulkan.
Part of the overnight 2026-07-04 llama.cpp Vulkan sweep (`results/overnight-20260704-070027/`).

- **Result (conc 32):** prefill 79.90 / decode **96.86** tok/s aggregate; 355/1000 prompts
  (hit the 900 s cap), **10 errors** (slot-split — 2048 tokens/request).
- **Prefill regresses vs c8** (79.90 vs 109.91) while decode is flat (96.86 vs 97.25): with 32
  slots contending, batched prefill scheduling loses more than the extra streams gain. c8 is
  strictly better for this model — see [`-c8`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c8).
- **Memory:** VRAM delta **22.86 GiB** (peak 24.59) — highest of the sweep, still under a quarter
  of the 96 GiB pool; MemAvailable delta 11.15 GB.
- Sweep siblings: [`-c1`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c1) ·
  [`-c8`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c8).
