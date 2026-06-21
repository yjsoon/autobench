---
title: Gemma 4 E4B · llama.cpp · Q4_K_M
model: google/gemma-4-E4B-it-qat-w4a16-ct
company: Google
family: Gemma
params: ~4B (edge)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance.
source_repo: google/gemma-4-E4B-it-qat-w4a16-ct
download_url: https://huggingface.co/google/gemma-4-E4B-it-qat-w4a16-ct
context: 131072
modalities: [text, image]
mm_served: true
tags: [Google, Gemma, Q4_K_M, ≤4B]

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

Edge ("E") Gemma 4; QAT, mobile-ready.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
