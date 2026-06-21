---
title: Qwen3.5-397B-A17B · llama.cpp · UD-IQ1_M
model: Qwen/Qwen3.5-397B-A17B
company: Alibaba
family: Qwen
params: 397B / 17B (MoE)
engine: llama.cpp
quant: UD-IQ1_M
quant_rationale: At ~1.7-bit, UD-IQ1_M (107 GB) is the ONLY quant of this 397B model that fits a single 128 GB Spark — and Unsloth's dynamic 1-bit keeps attention (MLA) + MoE-router layers at higher precision so it stays coherent instead of looping. (IQ2_XXS is 115 GB, IQ2_M 123 GB — too tight.)
source_repo: unsloth/Qwen3.5-397B-A17B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.5-397B-A17B-GGUF
context: 131072
modalities: [text]
mm_served: true
tags: [Alibaba, Qwen, UD-IQ1_M, 130B+]

status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
run_command: |
  # planned: llama.cpp --bench on the UD-IQ1_M GGUF; expect a tiny context (KV headroom is ~10–15 GB)
---

**Why this is interesting:** this is the single most "crazy-large" model that actually fits one
Spark. At 397B total it's normally 🔴 — but at Unsloth's dynamic ~1.7-bit (UD-IQ1_M, **107 GB**) the
weights drop under the 128 GB ceiling with ~10–15 GB to spare. Because it's **MoE with only 17B
active params**, decode stays usable (a dense 400B at the same size would crawl at ~5 tok/s).

It's a deliberate stress test of three things at once: the **memory ceiling** (how much context fits
on top of 107 GB of weights before OOM), the **sub-2-bit quant regime** (does dynamic 1-bit hold up),
and **large-MoE decode bandwidth** on the GB10. Record the max context that doesn't OOM as a result.

The 600B+/1T flagships (DeepSeek-R1 671B ≥162 GB at 1-bit, Kimi K2.6 1T ≥340 GB) do **not** fit even
at 1-bit — they stay multi-node. This 397B MoE is the boundary case.
