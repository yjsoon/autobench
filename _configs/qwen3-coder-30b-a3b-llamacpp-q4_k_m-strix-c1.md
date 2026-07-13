---
title: Qwen3-Coder 30B-A3B · llama.cpp · Q4_K_M · Strix Halo · conc 1
model: Qwen/Qwen3-Coder-30B-A3B-Instruct
company: Alibaba
family: Qwen
params: 30B / 3B (MoE)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
download_url: https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [qwen3-coder-30b-a3b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-1, strix-halo]
status: done
prefill_toks: 102.8
decode_toks: 73.67
mem_gb: 23.26
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 47.26 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 47.26
vram_delta_gb: 23.26
measured_on: 2026-07-14
completed_at: 2026-07-14 03:42:55 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 301/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The coding MoE (3B active) — fastest mid-tier decode of the sweep and directly relevant to the local OpenCode setup.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 102.8 / decode **73.67** tok/s;
  301/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 1294.38 / decode **96.83** tok/s.
- Spark ran Qwen3-Coder-30B-A3B on vLLM FP8 (decode 295.82) and the DDTree harness, not llama.cpp — no same-engine baseline.
- **Memory:** VRAM pool delta **23.26 GiB** (peak 47.26 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](qwen3-coder-30b-a3b-llamacpp-q4_k_m-strix-c8) · [`-c32`](qwen3-coder-30b-a3b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/qwen3-coder-30b-a3b-q4km/`.
