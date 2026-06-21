---
title: Granite-switch 4.1 30B · vLLM · FP8
model: ibm-granite/granite-switch-4.1-30b-preview
company: IBM
family: Granite
params: 32B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: ibm-granite/granite-switch-4.1-30b-preview
download_url: https://huggingface.co/ibm-granite/granite-switch-4.1-30b-preview
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

IBM MoE "switch" preview.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
