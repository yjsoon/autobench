---
title: gpt-oss-20b · llama.cpp · MXFP4 · Strix Halo · conc 32
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
concurrency: 32
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, conc-32, strix-halo]
status: done
prefill_toks: 164.22
decode_toks: 179.80
mem_gb: 7.97
mem_source: system MemAvailable delta (10s sampling); VRAM cross-check — delta 12.85 GiB (sysfs mem_info_vram_used, 96 GiB UMA pool; peak counter 35.41 GiB includes a co-resident idle llama-server holding ~22 GiB)
vram_delta_gb: 12.85
measured_on: 2026-07-06
completed_at: 2026-07-06 21:32 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
run_command: |
  # Vulkan (RADV) container, GPU via --device /dev/dri --device /dev/kfd.
  # Wrapper expands to: llama-server -ngl 99 -c 65536 --parallel 32 -cb + bench-serving.py (ShareGPT V3).
  # NOTE: llama.cpp splits -c across slots → only 2048 tokens per request at conc 32.
  scripts/bench-llamacpp-serving.sh \
    ggml-org/gpt-oss-20b-GGUF/gpt-oss-20b-mxfp4.gguf 65536 32 1000 900 256 99
  # 698/1000 prompts (hit 900 s cap), 15 errors (slot-split).
---

**gpt-oss-20b at conc 32 on Strix Halo** — decode keeps climbing past conc 8 (unlike
Qwen3.6-35B-A3B, which saturates there), but the gap to the Spark's batch throughput is widest
here. Runs on llama.cpp b9859, past the harmony blocker (see
[`-c1`](gpt-oss-20b-llamacpp-mxfp4-strix-c1) for the story).

- **Result (conc 32):** prefill 164.22 / decode **179.80** tok/s aggregate; 698/1000 prompts
  (hit the 900 s cap), **15 errors** (slot-split — 2048 tokens/request).
- **vs Spark vLLM MXFP4 conc 32** ([`gpt-oss-20b-vllm-mxfp4`](gpt-oss-20b-vllm-mxfp4):
  654.13 prefill / 535.29 decode): **~34% of Spark decode** — batch is where the Spark + vLLM
  combination runs away; at conc 1 the ranking flips
  ([`-c1`](gpt-oss-20b-llamacpp-mxfp4-strix-c1) beats the Spark by 50%).
- **Memory:** MemAvailable delta 7.97 GB (headline); VRAM delta **12.85 GiB** — the whole sweep
  never used more than a seventh of the 96 GiB pool.
- Sweep siblings: [`-c1`](gpt-oss-20b-llamacpp-mxfp4-strix-c1) ·
  [`-c8`](gpt-oss-20b-llamacpp-mxfp4-strix-c8). Evidence:
  `results/sweep-gptoss-20b-mxfp4-20260706-204546/`.
