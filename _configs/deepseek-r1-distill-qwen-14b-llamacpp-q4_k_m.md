---
title: DeepSeek-R1-Distill-Qwen-14B · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-14B
company: DeepSeek
family: DeepSeek
params: 14B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
download_url: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-14B-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [DeepSeek, Q4_K_M, 5-15B]

status: done
prefill_toks: 178.42
decode_toks: 243.75
mem_gb: 30.85
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 04:43 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model DeepSeek-R1-Distill-Qwen-14B-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The older R1 distill on a Qwen2.5-14B base — clean dense scaling, and another point against the
Gemma KV cliff.** Q4_K_M from bartowski.

- **Workload:** ShareGPT V3, concurrency 32. **982/1000, 18 errors** (slot-split) — completed all
  dispatched prompts; the run drained at **957 s** (just over the 900 s dispatch deadline as in-flight
  requests finished, so `hit_time_cap=false`).
- **Throughput (aggregate, conc 32):** prefill **178.4 tok/s**, decode **243.8 tok/s**. TTFT median
  **711 ms**, TPOT median **111.5 ms** (≈9 tok/s/stream), req throughput 1.03/s. Sits right where a
  dense 14B should: below the 8Bs (365–375) and above the compute-heavy Gemma-12B (195).
- **Memory: 30.9 GB — and that's the headline comparison.** A *14B* Qwen here uses **less** memory than
  the *12B* Gemma (41.3 GB). Same quant, same 64K ctx, same engine — the only difference is attention
  architecture. Qwen2.5's standard GQA KV is far cheaper than Gemma's wide global-attention KV, so the
  larger model has the smaller footprint. Two independent dense data points (this + Llama-3.1-8B) now
  confirm the Gemma-12B number is an architecture cliff, not a measurement fluke.
- **Reasoning model:** 233 k completion tokens over 982 reqs ≈ 237 each — long outputs near the 256
  cap, consistent with the R1 distill's think-then-answer behaviour.
- **Slot-split errors (18):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
