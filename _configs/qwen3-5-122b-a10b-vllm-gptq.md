---
title: Qwen3.5-122B-A10B · vLLM · GPTQ-Int4
model: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
company: Alibaba
family: Qwen
params: 122B / 10B (MoE)
engine: vLLM
quant: GPTQ-Int4
quant_rationale: 4-bit to fit one Spark; official GPTQ-Int4 weights published by the lab.
source_repo: Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
download_url: https://huggingface.co/Qwen/Qwen3.5-122B-A10B-GPTQ-Int4
context: 131072
modalities: [text]
mm_served: true
tags: [qwen3.5-122b-a10b, Alibaba, Qwen, GPTQ-Int4, 41-130B]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # superseded — replaced by the Qwen3.6 MoE (see Notes)
---

**Blocked — superseded by Qwen3.6.** Decision 2026-06-22.

Qwen released **Qwen3.6** (27B dense + **35B-A3B MoE**), which **outperforms the Qwen3.5 line** across
coding/agentic benchmarks at a fraction of the size — per Qwen's own results, even the 27B dense beats
the *397B* 3.5 model ([qwen.ai/blog](https://qwen.ai/blog?id=qwen3.6-27b)). So this 122B/10B 3.5 MoE is
no longer the best Qwen in its niche. Replaced in the queue by **`Qwen/Qwen3.6-35B-A3B-FP8`** (the
sparse-MoE 3.6 counterpart) — benchmarked as its own configs, base **and** with the model's **native
MTP** speculative module. Left blocked (not deleted) to preserve the decision trail; unblock only if a
direct 3.5-vs-3.6 large-MoE comparison is wanted.
