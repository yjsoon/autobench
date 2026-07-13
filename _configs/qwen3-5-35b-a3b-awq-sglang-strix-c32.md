---
title: Qwen3.5-35B-A3B · SGLang · AWQ · Strix Halo · conc 32
model: cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
company: Alibaba
family: Qwen
params: 35B / 3.3B (MoE, 256 experts top-8, hybrid attn+mamba)
engine: SGLang
speculative:
quant: AWQ
quant_rationale: cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit (compressed-tensors AWQ) — the AWQ-MoE checkpoint the strix-halo-sglang project verifies on gfx1151; GPTQ-on-MoE and NVFP4 are NVIDIA-only, so AWQ is the runnable 4-bit path for SGLang here. Nearest SGLang analogue to the Vulkan Qwen3.6-35B-A3B Q4_K_M for a cross-engine comparison (note the 3.5-vs-3.6 minor version gap).
source_repo: cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
download_url: https://huggingface.co/cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
context: 8192
modalities: [text]
mm_served: false
concurrency: 32
tags: [qwen3.5-35b-a3b, Alibaba, Qwen, AWQ, 16-40B, conc-32, strix-halo]
status: done
prefill_toks: 58.28
decode_toks: 46.83
mem_gb: 31.1
mem_source: GPU VRAM footprint (sysfs mem_info_vram_used), SGLang-only — vram_peak 55.11 GiB minus the ~24 GiB co-resident OpenCode llama-server. System MemAvailable delta is meaningless here (model loads into the 96 GiB UMA pool before the sampler baseline; delta 1.45 GB).
vram_peak_gb: 55.11
measured_on: 2026-07-13
completed_at: 2026-07-13 22:49 +0800
engine_image: strix-halo-sglang:dev (locally built from github.com/JeremiahM37/strix-halo-sglang; image id sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272)
run_command: |
  # ROCm/gfx1151, WARM TunableOp. Part of the c1/c8/c32 sweep launched once via sweep-sglang.sh.
  scripts/sweep-sglang.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit qwen35-a3b-awq-warm 8192 "1 8 32" \
    --mem-fraction-static 0.5 --max-total-tokens 32768 --max-mamba-cache-size 64 \
    --attention-backend triton --disable-cuda-graph
  # c32: 199/1000 prompts (hit 900 s cap), 0 errors.
---

**Decode plateaus by c32 — SGLang's batching tops out well under the Vulkan path.** From c8 to c32,
aggregate decode barely moves (44.3 → 46.8) while prefill climbs (45.3 → 58.3): the schedule fills
but the MoE Triton decode kernels are the ceiling.

- **Result (conc 32, warm TunableOp):** prefill 58.28 / decode **46.83** tok/s aggregate; 199/1000
  prompts (hit the 900 s cap), **0 errors**.
- **vs Vulkan llama.cpp** ([`qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c32`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c32),
  decode 96.86): SGLang reaches **48%** of Vulkan's aggregate decode — the closest it gets, still
  roughly half.
- **The headline finding:** on Strix Halo (gfx1151), community **SGLang ROCm trails the mature
  llama.cpp Vulkan path at every concurrency** (34% / 46% / 48% of decode at c1/c8/c32). This
  *inverts* the DGX Spark, where SGLang/vLLM dominate llama.cpp — a reminder that the engine
  ranking is hardware-and-maturity specific, not portable.
- **Memory:** SGLang footprint ≈ **31.1 GiB** peak (co-resident with the ~24 GiB OpenCode server;
  VRAM peak counter 55.11 GiB includes both).
- Sweep siblings: [`-c1`](qwen3-5-35b-a3b-awq-sglang-strix-c1) ·
  [`-c8`](qwen3-5-35b-a3b-awq-sglang-strix-c8). Evidence:
  `results/sglang-qwen35-a3b-awq-warm-20260713-215844/`.
