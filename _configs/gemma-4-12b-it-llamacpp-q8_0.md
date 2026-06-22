---
title: Gemma 4 12B · llama.cpp · Q8_0
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: llama.cpp
quant: Q8_0
quant_rationale: ggml-org Q8_0 — the top (near-lossless) rung of the 12B llama.cpp quant ladder. The 12B is gemma4_unified (vLLM/SGLang-blocked), so llama.cpp multi-quant is its only coverage axis.
source_repo: ggml-org/gemma-4-12B-it-GGUF
download_url: https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF
context: 65536
modalities: [text]
mm_served: false
tags: [Google, Gemma, Q8_0, 5-15B]

status: done
prefill_toks: 111.22
decode_toks: 153.02
mem_gb: 47.87
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 11:18 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-12B-it-Q8_0.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-12B-it-Q8_0.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The near-lossless top of the 12B llama.cpp quant ladder — and, predictably, the slowest of the
plain quants.** Google's Gemma-4-12B, ggml-org Q8_0.

- **Workload:** ShareGPT V3, concurrency 32. **602/1000, 13 errors**, **hit the 15-min cap**.
- **Throughput (aggregate, conc 32):** prefill **111.2 tok/s**, decode **153.0 tok/s**. TTFT median
  **3.6 s**, TPOT median **171 ms**.
- **Heaviest weights, slowest decode — the bandwidth-bound rule, top of the ladder:** Q8_0 (8.5 bits/wt)
  decodes **153** vs Q6_K's 159 and Q4_K_M's 195, and carries the largest footprint at **47.9 GB**
  (weights ≈ 12 GB + Gemma's global-attention KV). It's the quality-preservation choice, not the
  throughput choice; on this model the Q4→Q8 step costs ~21% decode and ~7 GB for marginal quality.
- **Context:** part of the 12B's llama.cpp-only matrix (Q4_K_M / Q6_K / Q8_0 / +Google-MTP) — the 12B
  is gemma4_unified and unservable on the stock vLLM/SGLang images.
