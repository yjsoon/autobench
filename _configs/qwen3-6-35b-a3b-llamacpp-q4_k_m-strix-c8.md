---
title: Qwen3.6-35B-A3B · llama.cpp · Q4_K_M · Strix Halo · conc 8
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
concurrency: 8
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-8, strix-halo]
status: done
prefill_toks: 109.91
decode_toks: 97.25
mem_gb: 8.62
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — peak 23.01 GiB, delta 21.28 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool) — the VRAM counter is the one that moved for the model itself
vram_peak_gb: 23.01
vram_delta_gb: 21.28
measured_on: 2026-07-04
completed_at: 2026-07-04 08:15 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 8 -cb + bench-serving.py (ShareGPT V3).
  # NOTE: llama.cpp splits -c across slots → 8192 tokens per request at conc 8.
  scripts/bench-llamacpp-serving.sh \
    lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf 65536 8 1000 900 256 99
  # 346/1000 prompts (hit 900 s cap), 0 errors.
---

**The throughput sweet spot for this model on Strix Halo** — best clean (0-error) aggregate decode
of the sweep. Part of the overnight 2026-07-04 llama.cpp Vulkan sweep
(`results/overnight-20260704-070027/`).

- **Result (conc 8):** prefill 109.91 / decode **97.25** tok/s aggregate; 346/1000 prompts
  (hit the 900 s cap), **0 errors**.
- **Concurrency scaling is shallow** for this MoE on Vulkan: 8× the streams buys only ~1.73× the
  decode of c1 (97.25 vs 56.08), and c32 adds nothing further (96.86, with errors) — decode is
  effectively saturated by c8.
- **Slot-split caveat:** 8192 tokens per request at conc 8 — too small for OpenCode on real repos,
  which is why the OpenCode server runs c1 despite this config's better aggregate throughput.
- **Memory:** VRAM delta **21.28 GiB** (peak 23.01); MemAvailable delta 8.62 GB.
- Sweep siblings: [`-c1`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c1) ·
  [`-c32`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c32).
