---
title: MiniMax-M2.7 · llama.cpp · UD-Q3_K_XL
model: MiniMaxAI/MiniMax-M2.7
company: MiniMax
family: MiniMax-M2
params: ~230B (≈10B active, MoE)
engine: llama.cpp
quant: UD-Q3_K_XL
quant_rationale: "Next smaller trusted Unsloth dynamic quant below the completed `UD-IQ4_XS` run. Verified from `unsloth/MiniMax-M2.7-GGUF` on 2026-06-23: `UD-Q3_K_XL` is about 101.9 GB vs `UD-IQ4_XS` at 108.4 GB, buying ~6.5 GB more headroom on the 121 GB GB10. Queue this next to see whether a slightly smaller runnable quant materially improves the pathological conc-32 result without dropping all the way to the tiny IQ2/IQ3 tiers."
source_repo: unsloth/MiniMax-M2.7-GGUF
download_url: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF/tree/main/UD-Q3_K_XL
context: 65536
modalities: [text]
concurrency: 32
tags: [minimax-m2-7, MiniMax, MiniMax-M2, Q3_K_XL, 130B+, conc-32]
status: done
prefill_toks: 65.22
decode_toks: 58.77
mem_gb: 117.62
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-24
completed_at: 2026-06-24 06:30 +0800
engine_image: ghcr.io/ggml-org/llama.cpp@sha256:12b288d6271e8de14412d61f641ca3ecd83bd73e1c4f4f22d86b2536f2b2f8e2
run_command: |
  HEALTH_TIMEOUT=900 scripts/bench-llamacpp-serving.sh \
    minimax-m2-7/UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf \
    65536 32 1000 900 256 99
---

**Smaller quant, bigger slot budget, basically the same overall story.** The original queued target was
128K, but after the interrupted attempt and missing-local-download recovery, the actual measured retry was
done at **64K (`65536`)** to avoid wasting another long startup on an obviously riskier context.

- **Load behavior:** `/health` came up after **319 s**, still a very expensive startup but notably better
  than the **446 s** seen on `UD-IQ4_XS`.
- **Serving result:** ShareGPT **conc-32**, target **1000 prompts / 900 s cap**. The run **hit the time
  cap** at **974.7 s** with **231 completed** and **7 HTTP 400s**. Aggregate request throughput was
  **0.237 req/s**.
- **Throughput:** prefill **65.22 tok/s**, decode **58.77 tok/s**. Prefill improved materially versus
  `UD-IQ4_XS` (**36.56 tok/s**), but decode is effectively unchanged (**58.42 -> 58.77 tok/s**). Median
  **TTFT 8095.6 ms**, median **TPOT 469.1 ms**.
- **Memory:** **117.62 GB**, essentially the same as `UD-IQ4_XS` (**117.78 GB**) despite ~6.5 GB smaller
  weights. In practice the recovered quant headroom gets eaten by the larger context/KV footprint.
- **What improved:** the HTTP 400 count fell from **29** on the `UD-IQ4_XS` `32768` run to **7** here,
  which fits the expected per-slot-budget story: `65536 / 32` gives about **2048 tokens per slot** versus
  about **1024** before.
- **Bottom line:** `UD-Q3_K_XL` is the better MiniMax llama.cpp rung than `UD-IQ4_XS` for conc-32
  serving, but it still does **not** turn this path into a practical ShareGPT benchmark on Spark. The
  dominant failure mode remains slow high-concurrency serving near the memory ceiling, not model-load
  instability.
