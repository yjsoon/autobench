---
title: Mistral Small 4 119B · vLLM · NVFP4
model: mistralai/Mistral-Small-4-119B-2603
company: Mistral AI
family: Mistral
params: 119B (dense-ish)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4 — hardware-accelerated on the GB10; first choice for NVIDIA models.
source_repo: mistralai/Mistral-Small-4-119B-2603
download_url: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603
context: 131072
modalities: [text]
mm_served: true
tags: [Mistral AI, Mistral, NVFP4, 41-130B]

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

NVFP4 variant exists; dense → slower decode.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
