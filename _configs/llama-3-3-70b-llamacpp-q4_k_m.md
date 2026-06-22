---
title: Llama 3.3 70B · llama.cpp · Q4_K_M
model: meta-llama/Llama-3.3-70B-Instruct
company: Meta
family: Llama
params: 70B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from unsloth (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: unsloth/Llama-3.3-70B-Instruct-GGUF
download_url: https://huggingface.co/unsloth/Llama-3.3-70B-Instruct-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [llama-3.3-70b, Meta, Llama, Q4_K_M, 41-130B, conc-32]
status: done
prefill_toks: 62.17
decode_toks: 48.7
mem_gb: 70.94
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 16:01 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/Llama-3.3-70B-Instruct-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model Llama-3.3-70B-Instruct-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
prefill_toks:
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The dense-70B tier opens the 41-130B bucket — roughly half the decode of the dense 32Bs, exactly as
the parameter count predicts.** Meta's Llama-3.3-70B, unsloth Q4_K_M.

- **Workload:** ShareGPT V3, concurrency 32. **273/1000, 6 errors**, **hit the 15-min cap** (drained at
  1205 s as the last in-flight batch finished). Loaded in 35 s.
- **Throughput (aggregate, conc 32):** prefill **62.2 tok/s**, decode **48.7 tok/s**. TTFT median
  **2.0 s**, TPOT median **476 ms** (≈2.1 tok/s/stream), req throughput 0.23/s.
- **Clean parameter scaling on llama.cpp.** A dense 70B fires ~2.2× the weights of the dense 32Bs per
  token, and decode lands at **48.7 vs the 32Bs' ~112** — almost exactly the inverse-parameter ratio.
  At conc 32 the GB10's ~273 GB/s of memory bandwidth is the wall: 70B of Q4 weights streamed per token
  caps throughput here. This is the slowest decode of any *non-reasoning* model so far, and a
  capability/fit test as much as a speed one.
- **Memory: 70.9 GB** — ~40 GB Q4_K_M weights + Llama's GQA KV at 64K ctx. Fits the 121 GB machine with
  headroom, unlike the Gemma-31B global-attention cliff (88 GB for half the parameters).
- **Slot-split errors (6):** `-c 65536 --parallel 32` → 2048 tok/slot. Low count only because so few
  requests completed before the cap.
- **For higher 70B-class throughput**, an FP8/AWQ build on vLLM (batched kernels) would beat llama.cpp
  here, the same pattern seen across the dense models — but the dense 70B is fundamentally
  bandwidth-bound on a single Spark.
