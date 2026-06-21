---
title: Llama 4 Scout 17B-16E · vLLM · AWQ-Int4
model: meta-llama/Llama-4-Scout-17B-16E-Instruct
company: Meta
family: Llama
params: 109B / 17B (MoE)
engine: vLLM
quant: AWQ-Int4
quant_rationale: 4-bit to fit one Spark; AWQ preserves quality well at Int4.
source_repo: meta-llama/Llama-4-Scout-17B-16E-Instruct
download_url: https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E-Instruct
context: 1048576
modalities: [text]
mm_served: true
tags: [Meta, Llama, AWQ-Int4, 41-130B]

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

~55 GB @4-bit; 10M-ctx claim → long-context test.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
