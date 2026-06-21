---
title: Granite 4.1 8B · llama.cpp · Q4_K_M
model: ibm-granite/granite-4.1-8b
company: IBM
family: Granite
params: 8B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance.
source_repo: ibm-granite/granite-4.1-8b
download_url: https://huggingface.co/ibm-granite/granite-4.1-8b
context: 131072
modalities: [text]
mm_served: true
tags: [IBM, Granite, Q4_K_M, 5-15B]

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

GGUF + FP8 official.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
