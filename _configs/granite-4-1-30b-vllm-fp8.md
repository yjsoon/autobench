---
title: Granite 4.1 30B · vLLM · FP8
model: ibm-granite/granite-4.1-30b
company: IBM
family: Granite
params: 28.9B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: ibm-granite/granite-4.1-30b
download_url: https://huggingface.co/ibm-granite/granite-4.1-30b
context: 131072
modalities: [text]
mm_served: true
tags: [IBM, Granite, FP8, 16-40B]

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

New IBM family; GGUF + FP8 official. Apache-2.0.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
