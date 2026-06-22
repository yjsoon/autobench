---
title: gpt-oss-120b · vLLM · MXFP4
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: vLLM
quant: MXFP4
quant_rationale: gpt-oss's native MXFP4 (~63 GB); FP4-accelerated on Blackwell with the CUTLASS kernels.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 131072
modalities: [text]
mm_served: true
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # filled in by the harness once the run completes
---

Flagship Spark-class test. Reported elsewhere at ~56–76 tok/s decode on a single
Spark; we'll record our own numbers here. Apache-2.0, 128K context.
