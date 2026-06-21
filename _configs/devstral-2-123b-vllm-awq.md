---
title: Devstral-2 123B · vLLM · AWQ-Int4
model: mistralai/Devstral-2-123B-Instruct-2512
company: Mistral AI
family: Devstral
params: 123B (MoE)
engine: vLLM
quant: AWQ-Int4
quant_rationale: 4-bit to fit one Spark; AWQ preserves quality well at Int4.
source_repo: mistralai/Devstral-2-123B-Instruct-2512
download_url: https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512
context: 131072
modalities: [text]
mm_served: true
tags: [Mistral AI, Devstral, AWQ-Int4, 41-130B]

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

Mistral flagship coder; replaces Codestral.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
