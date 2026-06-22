---
title: Qwen3-Coder-Next · vLLM · FP8
model: Qwen/Qwen3-Coder-Next
company: Alibaba
family: Qwen
params: 79.7B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: Qwen/Qwen3-Coder-Next
download_url: https://huggingface.co/Qwen/Qwen3-Coder-Next
context: 262144
modalities: [text]
mm_served: true
tags: [qwen3-coder-next, Alibaba, Qwen, FP8, 41-130B]
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

Best local coder line; FP8 + GGUF official. ~40 GB @4-bit.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
