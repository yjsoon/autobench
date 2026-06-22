---
title: Devstral-2 123B · vLLM · AWQ-Int4
model: mistralai/Devstral-2-123B-Instruct-2512
company: Mistral AI
family: Devstral
params: ~300B / 123B active (MoE)
engine: vLLM
quant: AWQ-Int4
quant_rationale: even 4-bit doesn't fit one Spark — see Notes.
source_repo: mistralai/Devstral-2-123B-Instruct-2512
download_url: https://huggingface.co/mistralai/Devstral-2-123B-Instruct-2512
context: 131072
modalities: [text]
mm_served: true
tags: [devstral-2-123b, Mistral AI, Devstral, AWQ-Int4, 41-130B]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — does not fit one Spark even at 4-bit (see Notes)
---

**Blocked — does not fit a single Spark, even at 4-bit.** Verified 2026-06-22.

The stub read "123B (MoE)", but **123B is the *active* count** — the model is a **~300B-total** MoE. The
only Int4 build that exists (`cpatonn/Devstral-2-123B-Instruct-2512-AWQ-4bit`, compressed-tensors
4-bit, group_size 32) is **~150 GB of safetensors** (confirmed: 300B params × ~4 bits ≈ 150 GB), which
**exceeds the 121 GB unified-memory ceiling** before any KV cache. The official `…-FP8` is ~300 GB —
far worse. So there is no quant of this model that loads on one GB10:

- 4-bit AWQ (cpatonn): ~150 GB → **OOM** (also a community quantizer, a secondary concern).
- FP8 (mistralai, official): ~300 GB → multi-node only.

**To fit, it would need ≤2-bit** (a GGUF UD-IQ1/IQ2 at ~75–95 GB, à la the Qwen3.5-397B fit
experiment) — if a *trusted* sub-3-bit GGUF appears, this could be revisited as a max-fit stress test
on llama.cpp. Until then it's out of reach on a single Spark. (Download was started, found to be
150 GB, and aborted; cache cleaned.)
