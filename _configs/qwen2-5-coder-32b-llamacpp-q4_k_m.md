---
title: Qwen2.5-Coder 32B · llama.cpp · Q4_K_M
model: Qwen/Qwen2.5-Coder-32B-Instruct
company: Alibaba
family: Qwen
params: 32B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: bartowski/Qwen2.5-Coder-32B-Instruct-GGUF
download_url: https://huggingface.co/bartowski/Qwen2.5-Coder-32B-Instruct-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [qwen2.5-coder-32b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-32]
status: done
prefill_toks: 108.21
decode_toks: 112.11
mem_gb: 45.95
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 13:21 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model Qwen2.5-Coder-32B-Instruct-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A strong dense 32B coder — closes the 16-40B bucket, and sits right with the other dense 32Bs on
llama.cpp.** Alibaba's Qwen2.5-Coder-32B, Q4_K_M from bartowski.

- **Workload:** ShareGPT V3, concurrency 32. **565/1000, 12 errors** (slot-split), **hit the 15-min
  cap**.
- **Throughput (aggregate, conc 32):** prefill **108.2 tok/s**, decode **112.1 tok/s**. TTFT median
  **790 ms**, TPOT median **230 ms** (≈4.3 tok/s/stream), req throughput 0.53/s.
- **Dense-32B-on-llama.cpp is a consistent tier:** decode **112** lands right beside the
  DeepSeek-R1-Distill-Qwen-32B (118) — same Qwen2.5-32B lineage, same engine/quant/ctx — and both are
  time-capped. A dense 32B fires all its parameters per token, so at conc 32 on llama.cpp it decodes in
  the ~110-120 band regardless of the fine-tune. The vLLM FP8 path (Qwen3-32B, 156) and especially the
  MoE/NVFP4 routes are where the 32B class gets faster.
- **Memory: 46.0 GB** — Qwen2.5 GQA KV at 64K ctx + the ~19 GB Q4_K_M weights; well clear of the
  Gemma-31B global-attention cliff (88 GB), in line with the other GQA 32Bs.
- **Slot-split errors (12):** `-c 65536 --parallel 32` → 2048 tok/slot. Engine-config artifact.
