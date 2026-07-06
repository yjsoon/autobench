---
title: gpt-oss-20b · llama.cpp · MXFP4 · Strix Halo · conc 8
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
concurrency: 8
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, conc-8, strix-halo]
status: done
prefill_toks: 154.45
decode_toks: 131.69
mem_gb: 9.20
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — delta 12.43 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool; peak counter 34.99 GiB includes a co-resident idle llama-server holding ~22 GiB)
vram_delta_gb: 12.43
measured_on: 2026-07-06
completed_at: 2026-07-06 21:16 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 8 -cb + bench-serving.py (ShareGPT V3).
  # NOTE: llama.cpp splits -c across slots → 8192 tokens per request at conc 8.
  scripts/bench-llamacpp-serving.sh \
    ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf 65536 8 1000 900 256 99
  # 495/1000 prompts (hit 900 s cap), 0 errors.
---

**gpt-oss-20b at conc 8 on Strix Halo** — clean scaling to ~1.9× the single-stream decode.
Runs on llama.cpp b9859, past the harmony chat-parser blocker that stopped the Spark's
llama.cpp attempt (see [`-c1`](gpt-oss-20b-llamacpp-mxfp4-strix-c1) for the story).

- **Result (conc 8):** prefill 154.45 / decode **131.69** tok/s aggregate; 495/1000 prompts
  (hit the 900 s cap), **0 errors**.
- **vs Spark vLLM MXFP4 conc 8** ([`gpt-oss-20b-vllm-mxfp4-c8`](gpt-oss-20b-vllm-mxfp4-c8):
  228.84 prefill / 212.52 decode): **~62% of Spark decode** — the single-stream edge
  ([`-c1`](gpt-oss-20b-llamacpp-mxfp4-strix-c1) is +50%) inverts once batching kicks in.
- **Slot-split caveat:** 8192 tokens per request at conc 8.
- **Memory:** MemAvailable delta 9.20 GB (headline); VRAM delta **12.43 GiB**.
- Sweep siblings: [`-c1`](gpt-oss-20b-llamacpp-mxfp4-strix-c1) ·
  [`-c32`](gpt-oss-20b-llamacpp-mxfp4-strix-c32). Evidence:
  `results/sweep-gptoss-20b-mxfp4-20260706-204546/`.
