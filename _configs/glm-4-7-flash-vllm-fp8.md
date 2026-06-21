---
title: GLM-4.7-Flash · vLLM · FP8
model: zai-org/GLM-4.7-Flash
company: Zhipu AI
family: GLM
params: 31B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: zai-org/GLM-4.7-Flash
download_url: https://huggingface.co/zai-org/GLM-4.7-Flash
context: 131072
modalities: [text]
mm_served: true
tags: [Zhipu AI, GLM, FP8, 16-40B]

status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — no confirmable repo (see Notes)
---

**Blocked — the named model does not resolve to a real HF repo, and no clean substitute fits the
intended slot.** Verified 2026-06-22.

- `zai-org/GLM-4.7-Flash` (and `glm-4-7-flash`, `GLM-4-Flash`) → **404** on HF. The stub describes a
  **~31B MoE GLM "Flash"**, which does not exist yet.
- The nearby real GLMs don't match: **`zai-org/GLM-4.5-Air`** is a **106B/12B MoE** (belongs in the
  41-130B / 130B+ buckets, not 16-40B), and **`zai-org/GLM-4-32B-0414`** is a **32B *dense*** model,
  not a Flash MoE. Substituting either would change the model class this config is meant to capture.
- Per the "**when unsure, BLOCK — don't guess**" policy, left for human review rather than silently
  swapped. If the intent is the dense 32B, point `source_repo` at `zai-org/GLM-4-32B-0414`; if it's the
  small MoE, use `zai-org/GLM-4.5-Air` (and re-bucket by size).
