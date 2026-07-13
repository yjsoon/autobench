---
title: Qwen3.5-35B-A3B · SGLang · AWQ · Strix Halo · conc 8
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
concurrency: 8
tags: [qwen3.5-35b-a3b, Alibaba, Qwen, AWQ, 16-40B, conc-8, strix-halo]
status: done
prefill_toks: 45.29
decode_toks: 44.32
mem_gb: 29.9
mem_source: GPU VRAM footprint (sysfs mem_info_vram_used), SGLang-only — vram_peak 53.89 GiB minus the ~24 GiB co-resident OpenCode llama-server. System MemAvailable delta is meaningless here (model loads into the 96 GiB UMA pool before the sampler baseline; delta 1.98 GB).
vram_peak_gb: 53.89
measured_on: 2026-07-13
completed_at: 2026-07-13 22:31 +0800
engine_image: strix-halo-sglang:dev (locally built from github.com/JeremiahM37/strix-halo-sglang; image id sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272)
run_command: |
  # ROCm/gfx1151, WARM TunableOp. Part of the c1/c8/c32 sweep launched once via sweep-sglang.sh.
  scripts/sweep-sglang.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit qwen35-a3b-awq-warm 8192 "1 8 32" \
    --mem-fraction-static 0.5 --max-total-tokens 32768 --max-mamba-cache-size 64 \
    --attention-backend triton --disable-cuda-graph
  # c8: 160/1000 prompts (hit 900 s cap), 0 errors.
---

**The batch case — where SGLang was *supposed* to win, and still doesn't here.** SGLang's
continuous batching does scale better than llama.cpp on this box (2.3× c1→c8 vs Vulkan's 1.7×),
but from such a low single-stream floor that it stays behind in absolute terms.

- **Result (conc 8, warm TunableOp):** prefill 45.29 / decode **44.32** tok/s aggregate; 160/1000
  prompts (hit the 900 s cap), **0 errors**.
- **vs Vulkan llama.cpp** ([`qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c8`](qwen3-6-35b-a3b-llamacpp-q4_k_m-strix-c8),
  decode 97.25): SGLang reaches **46%** of Vulkan's aggregate decode. The strix-halo-sglang repo
  reports SGLang beating **Ollama** 3.35× at 8 streams — but that is against Ollama's poor
  batching on an 80-token synthetic workload; against a `-cb` continuous-batching `llama-server`
  on ShareGPT, SGLang does not catch up within c1–c32.
- **Memory:** SGLang footprint ≈ **29.9 GiB** peak (co-resident with the ~24 GiB OpenCode server;
  VRAM peak counter 53.89 GiB includes both).
- **Caveats:** Qwen3.5 AWQ vs Qwen3.6 Q4_K_M (version + quant gap); same ShareGPT workload/caps as
  every config.
- Sweep siblings: [`-c1`](qwen3-5-35b-a3b-awq-sglang-strix-c1) ·
  [`-c32`](qwen3-5-35b-a3b-awq-sglang-strix-c32). Evidence:
  `results/sglang-qwen35-a3b-awq-warm-20260713-215844/`.
