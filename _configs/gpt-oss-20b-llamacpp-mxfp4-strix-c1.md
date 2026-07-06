---
title: gpt-oss-20b · llama.cpp · MXFP4 · Strix Halo · conc 1
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: llama.cpp
speculative:
quant: MXFP4
quant_rationale: OpenAI ships gpt-oss natively in MXFP4 — ggml-org's GGUF is the reference llama.cpp packaging, and the same format the Spark vLLM baseline serves, so the quant is matched across machines for once.
source_repo: ggml-org/gpt-oss-20b-GGUF
download_url: https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, conc-1, strix-halo]
status: done
prefill_toks: 108.23
decode_toks: 68.48
mem_gb: 9.40
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — delta 12.36 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool; peak counter 34.98 GiB includes a co-resident idle llama-server holding ~22 GiB)
vram_delta_gb: 12.36
measured_on: 2026-07-06
completed_at: 2026-07-06 21:01 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 1 -cb + bench-serving.py (ShareGPT V3).
  scripts/bench-llamacpp-serving.sh \
    ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf 65536 1 1000 900 256 99
  # 257/1000 prompts (hit 900 s cap), 0 errors.
---

**The config the Spark couldn't measure — and a single-stream upset.** The Spark's llama.cpp
attempt ([`gpt-oss-20b-llamacpp-mxfp4`](gpt-oss-20b-llamacpp-mxfp4)) is `status: blocked`: build
b9744's harmony chat parser 500'd on every `/v1/chat/completions` request. **Fixed by b9859**
(this image) — the chat endpoint returns clean content + usage, so the number exists now.

- **Result (conc 1):** prefill 108.23 / decode **68.48** tok/s; 257/1000 prompts (hit the 900 s
  cap), **0 errors**.
- **vs Spark vLLM MXFP4 conc 1** ([`gpt-oss-20b-vllm-mxfp4-c1`](gpt-oss-20b-vllm-mxfp4-c1):
  64.57 prefill / 45.56 decode): this Strix Halo llama.cpp run decodes **+50% faster** at single
  stream — same MXFP4 weights, different engine. The Spark wins back decisively at batch
  (see [`-c32`](gpt-oss-20b-llamacpp-mxfp4-strix-c32)).
- **Synthetic ceiling** (llama-bench pp512/tg128, same session): prefill 1672.4 ± 13.1 / decode
  **81.5 ± 0.2** tok/s — the fastest tg128 measured on this box so far (vs Qwen3.6-35B 76.2,
  Gemma E4B 61.2).
- **Memory:** MemAvailable delta 9.40 GB (headline); VRAM delta **12.36 GiB** for the 11.27 GiB
  GGUF + KV.
- Sweep siblings: [`-c8`](gpt-oss-20b-llamacpp-mxfp4-strix-c8) ·
  [`-c32`](gpt-oss-20b-llamacpp-mxfp4-strix-c32). Evidence:
  `results/sweep-gptoss-20b-mxfp4-20260706-204546/`.
