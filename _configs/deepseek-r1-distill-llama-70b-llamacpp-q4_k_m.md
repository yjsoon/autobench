---
title: DeepSeek-R1-Distill-Llama-70B · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
company: DeepSeek
family: DeepSeek
params: 70B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance.
source_repo: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
download_url: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
context: 131072
modalities: [text]
mm_served: true
tags: [DeepSeek, DeepSeek, Q4_K_M, 41-130B]

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

R1 distill on Llama-70B; dense 70B → ~5–8 tok/s, capability test.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
