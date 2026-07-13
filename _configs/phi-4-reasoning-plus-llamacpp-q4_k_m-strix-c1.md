---
title: Phi-4-reasoning-plus · llama.cpp · Q4_K_M · Strix Halo · conc 1
model: microsoft/Phi-4-reasoning-plus
company: Microsoft
family: Phi
params: 14B
engine: llama.cpp
speculative:
quant: Q4_K_M
quant_rationale: Q4_K_M GGUF from unsloth — the standard 4-bit llama.cpp packaging; matched to the Spark llama.cpp baseline quant where one exists.
source_repo: unsloth/Phi-4-reasoning-plus-GGUF
download_url: https://huggingface.co/unsloth/Phi-4-reasoning-plus-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [phi-4-reasoning-plus, Microsoft, Phi, Q4_K_M, 5-15B, conc-1, strix-halo]
status: done
prefill_toks: 51.71
decode_toks: 21.42
mem_gb: 20.85
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 44.99 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 44.99
vram_delta_gb: 20.85
measured_on: 2026-07-14
completed_at: 2026-07-14 00:33:46 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/Phi-4-reasoning-plus-GGUF/Phi-4-reasoning-plus-Q4_K_M.gguf 65536 1 1000 900 256 99
  # 76/1000 prompts (hit the 900 s cap), 0 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**A 14B reasoning model — prefill-heavy (fast prompt ingest), decode more modest.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 1):** prefill 51.71 / decode **21.42** tok/s;
  76/1000 prompts (hit the 900 s cap), **0 errors**.
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 770.44 / decode **24.71** tok/s.
- **Spark same-engine baseline:** [`phi-4-reasoning-plus-llamacpp-q4_k_m`](phi-4-reasoning-plus-llamacpp-q4_k_m) (llama.cpp CUDA Q4_K_M, conc 32: decode 230.99).
- **Memory:** VRAM pool delta **20.85 GiB** (peak 44.99 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c8`](phi-4-reasoning-plus-llamacpp-q4_k_m-strix-c8) · [`-c32`](phi-4-reasoning-plus-llamacpp-q4_k_m-strix-c32). Evidence: `results/batch-20260713-225032/phi-4-reasoning-plus-q4km/`.
