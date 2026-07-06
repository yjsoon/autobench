---
title: Qwen3.6-35B-A3B · llama.cpp · Q4_K_M · Strix Halo · conc 1
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
concurrency: 1
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-1, strix-halo]
status: done
prefill_toks: 69.51
decode_toks: 56.08
mem_gb: 8.51
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — peak 22.61 GiB, delta 20.88 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool) — the VRAM counter is the one that moved for the model itself
vram_peak_gb: 22.61
vram_delta_gb: 20.88
measured_on: 2026-07-04
completed_at: 2026-07-04 08:00 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh \
    lmstudio-community/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 198/1000 prompts (hit 900 s cap), 0 errors.
---

**The local OpenCode default configuration** — single slot keeps the full 65536-token context,
which is why this (not c8) is what `scripts/serve-qwen-opencode.sh` runs. Part of the overnight
2026-07-04 llama.cpp Vulkan sweep (`results/overnight-20260704-070027/`).

- **Result (conc 1):** prefill 69.51 / decode **56.08** tok/s; 198/1000 prompts (hit the 900 s cap),
  **0 errors**.
- **vs Spark** ([`qwen3-6-35b-a3b-nvfp4-vllm-c1`](qwen3-6-35b-a3b-nvfp4-vllm-c1), vLLM NVFP4 conc 1:
  93.25 prefill / 74.74 decode): Strix Halo reaches **~75% of Spark decode** — but note the quant gap
  (Q4_K_M GGUF vs NVFP4) and engine gap (llama.cpp Vulkan vs vLLM), so this is a
  platform-plus-stack comparison, not a chip-only one.
- **Memory:** the model lands in the 96 GiB UMA VRAM pool — VRAM delta **20.88 GiB** (peak 22.61)
  for the 19.70 GiB GGUF + KV; MemAvailable delta only 8.51 GB.
- **Synthetic ceiling** (same night, llama-bench pp512/tg128): prefill 1115.5 ± 15.0 / decode
  76.2 ± 0.6 tok/s (VRAM peak 20.98 GiB) — serving decode at c1 is ~74% of the synthetic tg128 rate.
- Sweep siblings: [`-c8`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c8) ·
  [`-c32`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c32).
