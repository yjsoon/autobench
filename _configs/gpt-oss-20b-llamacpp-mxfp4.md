---
title: gpt-oss-20b · llama.cpp · MXFP4
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: llama.cpp
quant: MXFP4
quant_rationale: gpt-oss's native FP4 format; the ggml-org GGUF is a single 12 GB MXFP4 file.
source_repo: ggml-org/gpt-oss-20b-GGUF
download_url: https://huggingface.co/ggml-org/gpt-oss-20b-GGUF
context: 131072
modalities: [text]
mm_served: true
concurrency: 32
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-06-21
completed_at:
run_command: |
  # Model loads & generates via raw /completion, but the OpenAI /v1/chat/completions
  # endpoint cannot be benchmarked — see below.
---

**BLOCKED — llama.cpp can't serve gpt-oss's harmony chat format on this build.**

The GGUF downloads and loads fine, and raw text completion works
(`/completion "The capital of France is"` → "Paris…"). But every request to the
**OpenAI chat endpoint** (`/v1/chat/completions`, which the ShareGPT harness and every
other engine use) fails mid-stream with:

> `500 — The model produced output that does not match the expected peg-native format`

This is llama.cpp's **harmony** response-format parser rejecting the model's channel
output. It persists with `--jinja` and `--reasoning-format none` (build `b9744-063d9c156`).
So no comparable OpenAI-chat throughput number can be produced on this engine right now.

This is consistent with the Spark guidance: **gpt-oss's recommended engines are SGLang
(claimed SOTA, ~70 tok/s decode on 20b) and vLLM**, not llama.cpp. The benchmark for
gpt-oss-20b will be produced on **vLLM / SGLang** (see the `gpt-oss-20b · vLLM · MXFP4`
config) once that engine is stood up. Revisit llama.cpp if a later build fixes harmony chat
parsing (raw `/completion` is a fallback but isn't OpenAI-chat-comparable and omits usage tokens).
