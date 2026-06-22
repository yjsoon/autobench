---
title: Gemma 4 31B · llama.cpp · Q4_K_M
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from ggml-org (trusted, llama.cpp's own org) — widest llama.cpp coverage.
source_repo: ggml-org/gemma-4-31B-it-GGUF
download_url: https://huggingface.co/ggml-org/gemma-4-31B-it-GGUF
context: 65536
modalities: [text, image]
mm_served: false
tags: [Google, Gemma, Q4_K_M, 16-40B]

status: done
prefill_toks: 67.78
decode_toks: 78.45
mem_gb: 88.12
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 08:29 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-31B-it-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-31B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The Gemma global-attention KV cliff at full force — the slowest, heaviest run in the entire
benchmark.** Google's dense Gemma-4-31B, Q4_K_M from ggml-org.

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at just **322/1000, 10 errors** —
  by far the fewest completed of any config.
- **Throughput (aggregate, conc 32):** prefill **67.8 tok/s**, decode **78.5 tok/s** — **the lowest of
  the whole benchmark.** TTFT median **13.0 s** (!), TPOT median **310 ms** (≈3.2 tok/s/stream), req
  throughput 0.31/s.
- **Memory: 88.1 GB — for a 31B at Q4_K_M (weights only ~18 GB).** This is the headline: Gemma's wide
  **global-attention KV cache** at 64K ctx × 32 slots balloons to ~70 GB on its own, pushing the
  footprint to 88 GB and leaving little bandwidth headroom. The same cliff seen on Gemma-4-12B (41 GB)
  scales brutally with depth/size — the 31B nearly saturates the 121 GB machine at this serving config
  and crawls. By contrast the dense Qwen2.5/DeepSeek 32Bs sit ~30 GB with GQA. **Architecture, not
  parameter count, decides the footprint — and here it decides usability.**
- **This is the strongest motivation in the set for an NVFP4 / vLLM build of Gemma-31B.** NVFP4 weights
  + vLLM's paged KV would cut both the weight traffic and the KV materialization dramatically; an
  `nvidia/Gemma-4-31B-IT-NVFP4` run is queued for approval as the fast-path comparison.
- **Slot-split errors (10):** `-c 65536 --parallel 32` → 2048 tok/slot. Engine-config artifact (low
  count here only because so few requests completed at all).

