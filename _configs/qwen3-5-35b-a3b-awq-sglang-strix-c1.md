---
title: Qwen3.5-35B-A3B · SGLang · AWQ · Strix Halo · conc 1
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
concurrency: 1
tags: [qwen3.5-35b-a3b, Alibaba, Qwen, AWQ, 16-40B, conc-1, strix-halo]
status: done
prefill_toks: 31.07
decode_toks: 19.03
mem_gb: 28.6
mem_source: GPU VRAM footprint (sysfs mem_info_vram_used), SGLang-only — vram_base 52.64 GiB minus the ~24 GiB co-resident OpenCode llama-server. System MemAvailable delta is meaningless here (model loads into the 96 GiB UMA pool before the sampler baseline; delta 1.57 GB).
vram_peak_gb: 52.96
measured_on: 2026-07-13
completed_at: 2026-07-13 22:15 +0800
engine_image: strix-halo-sglang:dev (locally built from github.com/JeremiahM37/strix-halo-sglang; image id sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272)
run_command: |
  # ROCm/gfx1151. WARM TunableOp cache (cold ≈ 40% slower — see note). Server launched once,
  # driven at c1/c8/c32 via scripts/sweep-sglang.sh + bench-serving.py (ShareGPT V3).
  scripts/sweep-sglang.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit qwen35-a3b-awq-warm 8192 "1 8 32" \
    --mem-fraction-static 0.5 --max-total-tokens 32768 --max-mamba-cache-size 64 \
    --attention-backend triton --disable-cuda-graph
  # c1: 67/1000 prompts (hit 900 s cap), 0 errors.
---

**First SGLang ROCm datapoint on this box — and the single-stream floor of the ROCm-vs-Vulkan
story.** SGLang on gfx1151 works (community `strix-halo-sglang` image, wave32 + AWQ-MoE-triton
patches), but single-stream it trails the mature Vulkan llama.cpp path badly.

- **Result (conc 1, warm TunableOp):** prefill 31.07 / decode **19.03** tok/s; 67/1000 prompts
  (hit the 900 s cap), **0 errors**.
- **vs Vulkan llama.cpp** ([`qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c1`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c1),
  decode 56.08): SGLang reaches only **34%** of Vulkan's single-stream decode. llama.cpp ships
  hand-tuned HIP MoE kernels and a near-zero dispatch path; SGLang's MoE Triton kernels on RDNA 3.5
  (no aiter Flash Attention yet) are the bottleneck.
- **TunableOp gotcha (load-bearing):** a *cold* cache gives **11.6 tok/s** here (the first,
  cache-populating run) — this warm number is 1.6× that. The `~/.cache/strix-halo-sglang-tunableop`
  mount is not optional, and the first run of any model is cold; the real datapoint is the second run.
- **Memory:** SGLang footprint ≈ **28.6 GiB** (AWQ weights ~23 + mamba SSM + KV pool) in the 96 GiB
  UMA pool, co-resident with the OpenCode server (~24 GiB). VRAM peak counter 52.96 GiB includes both.
- **Caveats:** model is Qwen3.5 (SGLang AWQ) vs Qwen3.6 (Vulkan GGUF) — minor version + AWQ-vs-Q4_K_M
  quant gap; both are the 35B-A3B hybrid MoE. Same ShareGPT workload/caps as every config here.
- Sweep siblings: [`-c8`](qwen3-5-35b-a3b-awq-sglang-strix-c8) ·
  [`-c32`](qwen3-5-35b-a3b-awq-sglang-strix-c32). Evidence:
  `results/sglang-qwen35-a3b-awq-warm-20260713-215844/`.
