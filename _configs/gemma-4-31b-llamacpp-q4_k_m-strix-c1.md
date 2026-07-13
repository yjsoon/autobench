---
title: Gemma 4 31B · llama.cpp · Q4_K_M · Strix Halo · conc 1
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 31B (dense)
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/gemma-4-31B-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 1
tags: [gemma-4-31b, Google, Gemma, Q4_K_M, 16-40B, conc-1, strix-halo]
status: done
prefill_toks: 17.33
decode_toks: 9.76
mem_gb: 23.46
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 47.46 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 47.46
vram_delta_gb: 23.46
measured_on: 2026-07-14
completed_at: 2026-07-14 04:30:16 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/gemma-4-31B-it-GGUF/gemma-4-31B-it-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 35/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**The pool workout: dense 31B at c32 pushed VRAM to 83.7 GiB — the first run to seriously use the 96 GiB pool (KV grows fast for a dense 31B at 32 slots).**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 17.33 / decode **9.76** tok/s;
  35/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 300.11 / decode **11.68** tok/s.
- **Spark same-engine baseline:** [`gemma-4-31b-it-llamacpp-q4_k_m`](gemma-4-31b-it-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 78.45).
- **Memory:** VRAM pool delta **23.46 GiB** (peak 47.46 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](gemma-4-31b-llamacpp-q4_k_m-strix-c8) · [`-c32`](gemma-4-31b-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/gemma-4-31b-q4km/`.
