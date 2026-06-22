---
title: Llama 3.1 8B · llama.cpp · Q4_K_M
model: meta-llama/Llama-3.1-8B-Instruct
company: Meta
family: Llama
params: 8B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from unsloth (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance. The canonical small dense baseline.
source_repo: unsloth/Meta-Llama-3.1-8B-Instruct-GGUF
download_url: https://huggingface.co/unsloth/Meta-Llama-3.1-8B-Instruct-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [llama-3.1-8b, Meta, Llama, Q4_K_M, 5-15B, conc-32]
status: done
prefill_toks: 348.32
decode_toks: 365.22
mem_gb: 22.66
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 03:59 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/Llama-3.1-8B-Instruct-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model Llama-3.1-8B-Instruct-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The canonical dense-8B baseline — and it quietly upset the hybrid.** Meta's Llama-3.1-8B-Instruct,
Q4_K_M from unsloth.

- **Workload:** ShareGPT V3, concurrency 32. **985/1000, 15 errors** (slot-split) in **589 s** —
  **did not hit the time cap**. Loaded in **17 s**.
- **Throughput (aggregate, conc 32):** prefill **348.3 tok/s**, decode **365.2 tok/s**. TTFT median
  **234 ms**, TPOT median **76.2 ms** (≈13 tok/s/stream), req throughput 1.67/s.
- **The surprise: this dense 8B beat the Granite-4.1-8B hybrid on BOTH axes.** Decode **365 vs 324**,
  and memory **22.66 GB vs 25.41 GB** — the opposite of what the 3B comparison predicted. The reason
  is the serving shape: at `-c 65536 --parallel 32` every slot holds only **2048 tokens**, so the
  dense transformer's KV cache per stream is tiny and its quadratic-KV disadvantage never materializes.
  The Mamba-2 hybrid's win comes from cheap KV at *long* context; at 2048 tok/slot there's nothing to
  save, and the SSM scan carries its own overhead. **Granite's hybrid edge is real but context-length
  dependent — it does not show up under this short-per-slot 32-way workload, where the well-optimized
  dense Llama kernel simply wins.** (At the 3B tier Granite's 617 still led, so size + arch interact;
  the clean lesson here is that "hybrid always uses less memory" is false at this serving config.)
- **Slot-split errors (15):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
