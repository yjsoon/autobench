---
title: Phi-4-reasoning-plus · llama.cpp · Q4_K_M · Strix Halo · conc 32
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
concurrency: 32
tags: [phi-4-reasoning-plus, Microsoft, Phi, Q4_K_M, 5-15B, conc-32, strix-halo]
status: done
prefill_toks: 198.47
decode_toks: 131.38
mem_gb: 20.79
mem_source: GPU VRAM pool delta (sysfs mem_info_vram_used) — peak 44.93 GiB incl. the co-resident ~24 GiB OpenCode server; system MemAvailable barely moves (UMA pool)
vram_peak_gb: 44.93
vram_delta_gb: 20.79
measured_on: 2026-07-14
completed_at: 2026-07-14 01:05:22 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd. Part of the overnight
  # mid/small ladder batch (batch-sweep.sh + models-midsmall.tsv). Wrapper expands to:
  # llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh unsloth/Phi-4-reasoning-plus-GGUF/Phi-4-reasoning-plus-Q4_K_M.gguf 65536 32 1000 900 256 99
  # 508/1000 prompts (hit the 900 s cap), 12 errors. Batch driver: scripts/batch-sweep.sh scripts/models-midsmall.tsv
---

**Peak concurrency — where the slot-split (2048 tokens/request) starts costing errors.**
Part of the overnight 2026-07-13 mid/small ladder Vulkan sweep (`results/batch-20260713-225032/`).

- **Result (conc 32):** prefill 198.47 / decode **131.38** tok/s aggregate;
  508/1000 prompts (hit the 900 s cap), **12 errors** (slot-split — 2048 tokens/request).
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 770.44 / decode **24.71** tok/s.
- **vs Spark** ([`phi-4-reasoning-plus-llamacpp-q4_k_m`](phi-4-reasoning-plus-llamacpp-q4_k_m), llama.cpp CUDA Q4_K_M c32 decode 230.99): Strix Halo Vulkan reaches **57%** on the identical engine/quant/workload.
- **Memory:** VRAM pool delta **20.79 GiB** (peak 44.93 GiB, co-resident with the OpenCode server).
- Sweep siblings: [`-c1`](phi-4-reasoning-plus-llamacpp-q4_k_m-strix-c1) · [`-c8`](phi-4-reasoning-plus-llamacpp-q4_k_m-strix-c8). Evidence: `results/batch-20260713-225032/phi-4-reasoning-plus-q4km/`.
