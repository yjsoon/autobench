---
title: Gemma 4 E4B · llama.cpp · Q4_K_M
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from ggml-org (trusted, llama.cpp's own org). Stub pointed at the QAT-w4a16 checkpoint, but ggml-org publishes a clean GGUF — preferred for the llama.cpp path.
source_repo: ggml-org/gemma-4-E4B-it-GGUF
download_url: https://huggingface.co/ggml-org/gemma-4-E4B-it-GGUF
context: 65536
modalities: [text, image]
mm_served: false
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B]
status: done
prefill_toks: 328.95
decode_toks: 435.01
mem_gb: 16.76
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 05:27 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-E4B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The edge ("E") Gemma — and proof the Gemma memory cliff is a big-dense-model trait, not a family
trait.** Google's Gemma 4 E4B, an *elastic / MatFormer* ~4B-effective model, Q4_K_M from ggml-org.
Closes out the ≤4B bucket.

- **Workload:** ShareGPT V3, concurrency 32. **980/1000, 20 errors** (slot-split) in **563 s** —
  **no time cap**.
- **Throughput (aggregate, conc 32):** prefill **329.0 tok/s**, decode **435.0 tok/s** — faster decode
  than every dense 8B (365–375) and ~2.2× the dense Gemma-4-12B (195) on the identical engine/quant/ctx.
  TTFT median **763 ms**, TPOT median **59.7 ms** (≈17 tok/s/stream), req throughput 1.74/s.
- **Memory: 16.8 GB — the lightest of the whole small sweep, tied with the Granite-3B Mamba hybrid.**
  This directly qualifies the earlier "Gemma KV cliff" finding (Gemma-4-12B hit 41.3 GB): that cliff is
  about the *large dense* Gemma configs' wide global-attention KV, **not** the Gemma family as such. The
  edge E-series uses a narrow, KV-light configuration, so it sits at the bottom of the memory table, not
  the top. Architecture (and the specific attention config) decides footprint — restated from the other
  direction.
- **Quant note:** the stub pointed at Google's QAT-w4a16 checkpoint; ggml-org publishes a clean
  Q4_K_M GGUF, which is the right artifact for the llama.cpp path (the QAT-w4a16 is a vLLM/transformers
  format). Text path benchmarked; image input not served here (`mm_served: false`).
- **Slot-split errors (20):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
