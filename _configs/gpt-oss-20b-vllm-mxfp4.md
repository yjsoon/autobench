---
title: gpt-oss-20b · vLLM · MXFP4
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: vLLM
quant: MXFP4
quant_rationale: gpt-oss's native FP4 format; FP4-accelerated with the CUTLASS kernels.
source_repo: openai/gpt-oss-20b
download_url: https://huggingface.co/openai/gpt-oss-20b
context: 131072
modalities: [text]
mm_served: true
tags: [OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe]

status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned configuration — filled in by the run when benchmarked
---

Fast, Apache-2.0, native MXFP4.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
