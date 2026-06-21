---
title: Granite-switch 4.1 30B · vLLM · FP8
model: ibm-granite/granite-switch-4.1-30b-preview
company: IBM
family: Granite
params: 32B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: ibm-granite/granite-switch-4.1-30b-preview
download_url: https://huggingface.co/ibm-granite/granite-switch-4.1-30b-preview
context: 131072
modalities: [text]
mm_served: true
tags: [IBM, Granite, FP8, 16-40B]

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

**Blocked — no confirmable "switch" Granite repo.** Verified 2026-06-22.

- `ibm-granite/granite-switch-4.1-30b-preview` → **404** on HF; no IBM "switch"-named MoE resolves.
- The real ~30B Granite MoE is **`ibm-granite/granite-4.1-30b`** — already benchmarked as its own
  config (`granite-4-1-30b-vllm-fp8`, 221/182 tok/s). There is also `ibm-granite/granite-4.0-h-small`
  (a hybrid), but neither is a distinct "switch" architecture, so running one here would just duplicate
  an existing config under a wrong name.
- Per "**when unsure, BLOCK — don't guess**," left for human review. If IBM ships a genuine
  switch-MoE Granite, point `source_repo` at it; otherwise this config is a duplicate of
  granite-4.1-30b and can be dropped.
