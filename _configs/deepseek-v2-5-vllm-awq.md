---
title: DeepSeek-V2.5 · vLLM · AWQ-Int4
model: deepseek-ai/DeepSeek-V2.5
company: DeepSeek
family: DeepSeek
params: 236B / 21B (MoE)
engine: vLLM
quant: AWQ-Int4
quant_rationale: 236B MoE only fits at 4-bit (~118 GB weights) — and that leaves little headroom for KV cache against the 128 GB ceiling. Blocked pending a fit check.
source_repo: deepseek-ai/DeepSeek-V2.5
download_url: https://huggingface.co/deepseek-ai/DeepSeek-V2.5
context: 131072
modalities: [text]
mm_served: true
concurrency: 32
tags: [deepseek-v2.5, DeepSeek, AWQ-Int4, 130B+, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
run_command: |
  # blocked — not yet run
---

**Blocked for review.** 236B-total MoE: at 4-bit the weights alone are ~118 GB, leaving only
~10 GB for the CUDA context + KV cache on the 128 GB box — likely OOM at any real context length.
Worth a human decision on whether to attempt it (e.g. a more aggressive quant, tiny context, or
skip until two Sparks are linked). 21B active params would otherwise give usable decode speed.
