---
title: Qwen3.6-35B-A3B · vLLM · FP8
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 36B / 3B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: Qwen/Qwen3.6-35B-A3B
download_url: https://huggingface.co/Qwen/Qwen3.6-35B-A3B
context: 131072
modalities: [text]
mm_served: true
tags: [Alibaba, Qwen, FP8, 16-40B]

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

Newest mid Qwen MoE; very fast. FP8 variant published.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
