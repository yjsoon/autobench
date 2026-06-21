---
title: Llama 4 Maverick 402B · llama.cpp · UD-Q2_K_XL
model: meta-llama/Llama-4-Maverick-17B-128E-Instruct
company: Meta
family: Llama
params: 402B / 17B (MoE)
engine: llama.cpp
quant: UD-Q2_K_XL
quant_rationale: 2-bit (UD-Q2_K_XL, ~122 GB) is the smallest published quant. It technically fits the 128 GB box but leaves essentially no room for the CUDA context + KV cache, so it would OOM at any real context length — hence blocked for a human fit decision rather than a guaranteed run.
source_repo: unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
download_url: https://huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF
context: 131072
modalities: [text]
mm_served: true
tags: [Meta, Llama, UD-Q2_K_XL, 130B+]

status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
run_command: |
  # blocked — ~122 GB weights leave ~no KV headroom on 128 GB; needs a fit decision
---

**Why this is interesting (and why it's blocked):** a 402B MoE (17B active) running on one tiny box
is a great headline — at Unsloth's dynamic 2-bit it's **~122 GB**, which *just* fits 128 GB. The
catch is there's almost nothing left for KV cache, so it only runs at a trivially small context and
risks OOM. It's right at the edge: worth a human call on whether to attempt it (smaller-than-2-bit
quant, minimal context) or skip. Compare with the Qwen3.5-397B-A17B · UD-IQ1_M config, which fits
the same class of model with real headroom at ~1.7-bit.
