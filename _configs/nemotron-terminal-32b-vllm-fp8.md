---
title: Nemotron-Terminal-32B · vLLM · FP8
model: nvidia/Nemotron-Terminal-32B
company: NVIDIA
family: Nemotron
params: 33B (dense)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: nvidia/Nemotron-Terminal-32B
download_url: https://huggingface.co/nvidia/Nemotron-Terminal-32B
context: 131072
modalities: [text]
mm_served: true
tags: [NVIDIA, Nemotron, FP8, 16-40B]

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

NVIDIA agentic/terminal-tuned (Feb 2026).

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
