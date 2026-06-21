---
title: Nemotron-3 Nano-Omni 30B-A3B · vLLM · NVFP4
model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
company: NVIDIA
family: Nemotron
params: 33B / 3B (MoE)
engine: vLLM
quant: NVFP4
context: 131072
modalities: [text, image]
mm_served: true
tags: [NVIDIA, Nemotron, NVFP4, 16-40B, Spark recipe]

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

Best Spark showcase: ~56 tok/s decode, 7417 prefill (vLLM). Omni — verify audio/video support at download.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
