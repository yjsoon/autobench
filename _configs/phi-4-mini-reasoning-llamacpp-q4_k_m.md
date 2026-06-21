---
title: Phi-4-mini-reasoning · llama.cpp · Q4_K_M
model: microsoft/Phi-4-mini-reasoning
company: Microsoft
family: Phi
params: 3.8B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance.
source_repo: microsoft/Phi-4-mini-reasoning
download_url: https://huggingface.co/microsoft/Phi-4-mini-reasoning
context: 131072
modalities: [text]
mm_served: true
tags: [Microsoft, Phi, Q4_K_M, ≤4B]

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

Best sub-4B reasoner; ~3.5 GB @Q4. Smoke-test candidate.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
