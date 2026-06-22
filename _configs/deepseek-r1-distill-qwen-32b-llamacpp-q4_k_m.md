---
title: DeepSeek-R1-Distill-Qwen-32B · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
company: DeepSeek
family: DeepSeek
params: 32B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
download_url: https://huggingface.co/bartowski/DeepSeek-R1-Distill-Qwen-32B-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [DeepSeek, Q4_K_M, 16-40B]

status: done
prefill_toks: 84.23
decode_toks: 117.59
mem_gb: 51.55
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 08:49 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model DeepSeek-R1-Distill-Qwen-32B-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A dense 32B reasoning distill — slow and time-capped on llama.cpp, as expected for the size + long
reasoning outputs.** DeepSeek's R1-Distill on a Qwen2.5-32B base, Q4_K_M from bartowski.

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **534/1000, 10 errors**
  (slot-split).
- **Throughput (aggregate, conc 32):** prefill **84.2 tok/s**, decode **117.6 tok/s**. TTFT median
  **1.94 s**, TPOT median **211 ms** (≈4.7 tok/s/stream), req throughput 0.49/s.
- **Size + reasoning, compounding.** A dense 32B already taxes decode (all 32B fire per token); on top
  of that it's a reasoning model emitting long traces (127 k completion tokens over 534 reqs ≈ 238
  each, near the 256 cap), so few requests finish before the wall. Decode (118) lands between the dense
  Qwen3-32B FP8 on vLLM (156) and the heavier paths — slower here partly because Q4_K_M on llama.cpp at
  conc 32 doesn't match vLLM's batched FP8 kernels for a dense model this size.
- **Memory: 51.6 GB** — Qwen2.5's GQA KV at 64K ctx plus the ~19 GB Q4_K_M weights. Heavier than the
  other dense 32Bs' weights alone, but nowhere near the Gemma-31B global-attention cliff (88 GB) — GQA
  vs global attention, again.
- **Slot-split errors (10):** `-c 65536 --parallel 32` → 2048 tok/slot. Engine-config artifact.
