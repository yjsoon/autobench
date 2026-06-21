---
title: Nemotron-3 Elastic 30B-A3B · vLLM · NVFP4
model: nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4
company: NVIDIA
family: Nemotron
params: 30B / 3B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4 — hardware-accelerated on the GB10; first choice for NVIDIA models.
source_repo: nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4
context: 131072
modalities: [text]
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

Elastic-width MoE; NVFP4-native.

Stub — not yet benchmarked. Verify the exact HF repo ID, license, and quant availability at
download time (some mid-2026 names from `notes/MODELLIST.md` are still being confirmed).
