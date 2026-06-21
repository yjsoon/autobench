---
title: DeepSeek-V2-Lite-Chat · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-V2-Lite-Chat
company: DeepSeek
family: DeepSeek
params: 15.7B / 2.4B (MoE)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M — widest llama.cpp coverage, strong size/quality balance.
source_repo: deepseek-ai/DeepSeek-V2-Lite-Chat
download_url: https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat
context: 163840
modalities: [text]
mm_served: true
tags: [DeepSeek, DeepSeek, Q4_K_M, 16-40B]

status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — no top-trusted GGUF + architectural duplicate (see Notes)
---

**Blocked — two reasons, both pointing the same way.** Verified 2026-06-22.

1. **No top-trusted GGUF.** `bartowski`, `lmstudio-community`, and `QuantFactory`
   DeepSeek-V2-Lite-Chat-GGUF all **404**. Only `second-state/…` and `gaianet/…` publish one — known
   redistributors, but outside the ggml-org / unsloth / bartowski tier the policy calls "trusted." Per
   "**when unsure, BLOCK**," not run from a mid-tier source without review.
2. **Architectural near-duplicate of an already-measured config.** DeepSeek-V2-Lite-Chat is the same
   **15.7B / 2.4B-active MLA + 64-expert MoE** as **DeepSeek-Coder-V2-Lite-Instruct**, which *was*
   benchmarked (`deepseek-coder-v2-lite-instruct-llamacpp-q4_k_m`) and showed the slow llama.cpp
   MLA+fine-grained-MoE path on GB10 — **130 decode, time-capped, 38 GB**. A V2-Lite-Chat run would
   almost certainly reproduce that, adding little.

If a benchmark is wanted anyway, the chat-tuned result would be most informative on **vLLM** (to
separate engine from architecture), and/or approve the `second-state` GGUF for the llama.cpp path.
