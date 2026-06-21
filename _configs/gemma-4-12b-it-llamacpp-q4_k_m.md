---
title: Gemma 4 12B · llama.cpp · Q4_K_M
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from ggml-org (trusted, llama.cpp's own org) — widest llama.cpp coverage, strong size/quality balance.
source_repo: ggml-org/gemma-4-12B-it-GGUF
download_url: https://huggingface.co/ggml-org/gemma-4-12B-it-GGUF
context: 65536
modalities: [text, image]
mm_served: false
tags: [Google, Gemma, Q4_K_M, 5-15B]

status: done
prefill_toks: 141.11
decode_toks: 195.25
mem_gb: 41.28
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 04:27 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-12B-it-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-12B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The first 5-15B run to hit the time cap — and a memory cliff well above its parameter count.**
Google's Gemma 4 12B (text-only path here), Q4_K_M from ggml-org.

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **742/1000, 16 errors**
  (slot-split) — the first sub-1000 finish of the 5-15B set.
- **Throughput (aggregate, conc 32):** prefill **141.1 tok/s**, decode **195.3 tok/s** — roughly half
  the dense 8Bs (365–375) for only ~1.5× the parameters. TTFT median **2071 ms**, TPOT median
  **135.3 ms** (≈7.4 tok/s/stream), req throughput 0.77/s. Gemma is compute-heavy per token; the high
  TTFT is slow prefill (141 tok/s) backing up the queue at conc 32.
- **Memory cliff: 41.3 GB — nearly double the dense 8Bs (~23 GB) for a model only 50% larger.** The
  driver is Gemma's KV: a large `head_dim` and wide attention, with much of the network running
  **global (full) attention** rather than the cheap sliding-window layers, so the 64K-context KV cache
  pre-allocates far more than a Llama/Qwen 8B. This is the clearest "architecture decides footprint,
  not parameter count" data point so far on the dense side — and it's the opposite lesson from the
  Mamba hybrid: Gemma's attention pattern makes it *expensive* at long context.
- **Multimodal note:** the GGUF text path is benchmarked here; image input is not served via this
  llama-server config (`mm_served: false`). Vision throughput would need the mmproj projector and a
  separate run.
- **Slot-split errors (16):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
