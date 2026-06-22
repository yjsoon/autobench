---
title: Llama 3.3 70B · llama.cpp · Q4_K_M
model: meta-llama/Llama-3.3-70B-Instruct
company: Meta
family: Llama
params: 70B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance.
source_repo: meta-llama/Llama-3.3-70B-Instruct
download_url: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
context: 131072
modalities: [text]
mm_served: true
tags: [llama-3.3-70b, Meta, Llama, Q4_K_M, 41-130B]
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

Dense baseline; ~5–8 tok/s on Spark — capability test only.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
