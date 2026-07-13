---
title: Qwen3.6 27B · llama.cpp · Q4_K_M · Strix Halo · conc 1
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/Qwen3.6-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-GGUF
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-1, strix-halo]
status: done
prefill_toks: 20.6
decode_toks: 10.98
mem_gb: 19.33
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 43.33 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 43.33
vram_delta_gb: 19.33
measured_on: 2026-07-14
completed_at: 2026-07-14 02:56:12 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 39/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The dense-vs-MoE contrast: at 27B dense it decodes ~11 tok/s at c1, vs ~74 for the 30B-A3B MoE — active-parameter count, not total, sets single-stream speed.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 20.6 / decode **10.98** tok/s;
  39/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 361.82 / decode **12.77** tok/s.
- Spark ran Qwen3.6-27B only on vLLM/SGLang (NVFP4/FP8), not llama.cpp — no same-engine baseline; see the [model tag](../tags/model/) for the Spark configs.
- **Memory:** VRAM pool delta **19.33 GiB** (peak 43.33 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](qwen3-6-27b-llamacpp-q4_k_m-strix-c8) · [`-c32`](qwen3-6-27b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/qwen3p6-27b-q4km/`.
