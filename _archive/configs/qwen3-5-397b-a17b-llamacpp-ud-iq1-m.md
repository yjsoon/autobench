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
concurrency: 32
tags: [qwen3.5-397b-a17b, Alibaba, Qwen, UD-IQ1_M, 130B+, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
run_command: |
  # superseded on quality by Qwen3.6-27B; kept blocked as an optional max-fit stress test (see Notes)
---

**Blocked — superseded on quality by Qwen3.6, but flagged for review because it has a *separate*
value.** Decision 2026-06-22.

- **Quality supersession (the reason it's blocked):** Qwen's **Qwen3.6-27B** — a *27B dense* model —
  **beats this 397B/17B 3.5 model on every major coding benchmark** (SWE-bench Verified 77.2 vs 76.2,
  SWE-bench Pro 53.5 vs 50.9, Terminal-Bench 2.0 59.3 vs 52.5, SkillsBench 48.2 vs 30.0;
  [qwen.ai/blog](https://qwen.ai/blog?id=qwen3.6-27b)). There is **no Qwen3.6 giant** — the line tops out
  at 35B-A3B — so the "best Qwen" is now small. Per the replace-3.5-with-better-3.6 directive, this is
  replaced in the queue by **`Qwen3.6-27B-FP8`** (base + native MTP).
- **But this config also tested something Qwen3.6 can't:** the **max-fit boundary** — the single largest
  model (397B at Unsloth dynamic ~1.7-bit, 107 GB) that fits one 128 GB Spark, the sub-2-bit quant
  regime, and large-MoE decode bandwidth near the memory ceiling. That's a *fit/throughput* experiment,
  orthogonal to quality. **Unblock if you want the extreme-fit stress test** as its own data point;
  it's blocked only because the *quality* rationale that originally justified it is now obsolete.
