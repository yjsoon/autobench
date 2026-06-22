---
title: Qwen3.5-122B-A10B · vLLM · GPTQ-Int4
model: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
company: Alibaba
family: Qwen
params: 122B / 10B (MoE)
engine: vLLM
quant: GPTQ-Int4
quant_rationale: 4-bit to fit one Spark; official GPTQ-Int4 weights published by the lab.
source_repo: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
download_url: https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
context: 131072
modalities: [text]
mm_served: true
tags: [qwen3.5-122b-a10b, Alibaba, Qwen, GPTQ-Int4, 41-130B]
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

~61 GB; A10B active → fast decode. Int4 published by Qwen.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
