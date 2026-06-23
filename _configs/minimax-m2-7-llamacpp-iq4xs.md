---
title: MiniMax-M2.7 · llama.cpp · UD-IQ4_XS
model: MiniMaxAI/MiniMax-M2.7
company: MiniMax
family: MiniMax-M2
params: ~230B (≈10B active, MoE)
engine: llama.cpp
quant: UD-IQ4_XS
quant_rationale: unsloth dynamic IQ4_XS GGUF — the largest ≈4-bit MiniMax-M2.7 quant that FITS the 121 GB GB10 (108.4 GB weights, ~12 GB headroom for KV). The requested MXFP4_MOE (136 GB) and saricles NVFP4 (130 GB) both exceed the box — see those configs. This is the runnable 4-bit stand-in from the same trusted unsloth repo.
source_repo: unsloth/MiniMax-M2.7-GGUF
download_url: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF/tree/main/UD-IQ4_XS
context: 32768
modalities: [text]
concurrency: 32
tags: [minimax-m2-7, MiniMax, MiniMax-M2, IQ4_XS, 130B+, conc-32]
status: done
prefill_toks: 36.56
decode_toks: 58.42
mem_gb: 117.78
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-23
completed_at: 2026-06-23 22:37 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:12b288d6271e8de14412d61f641ca3ecd83bd73e1c4f4f22d86b2536f2b2f8e2
run_command: |
  # llama-server (NGC dispatcher image) + ShareGPT serving benchmark, conc=32
  docker run --rm --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/minimax-m2-7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    -ngl 99 -c 32768 --parallel 32 -cb \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model minimax-m2-7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

Runnable 4-bit MiniMax-M2.7 stand-in for the blocked MXFP4/NVFP4 requests — this is the largest
trusted GGUF in the unsloth repo that actually fits the 121 GB GB10.

- **Workload:** ShareGPT V3, conc-32, target 1000 prompts / 15 min cap. The run **hit the time cap**
  at **997.3 s** with only **241 completed** and **29 HTTP 400 errors**, so this path is far too slow
  for the standard workload at this batch size. Aggregate request throughput was **0.242 req/s**.
- **Throughput:** prefill **36.56 tok/s**, decode **58.42 tok/s**. Median **TTFT 4348 ms**, median
  **TPOT 488.5 ms** (about 2.0 tok/s per live stream once generation starts).
- **Memory:** **117.78 GB**, effectively the whole box. With **108.4 GB** of weights already resident,
  MiniMax-M2.7 leaves almost no headroom for KV/cache growth; the 32768 context was necessary just to
  fit conc-32 at all.
- **Load behavior:** `/health` took **446 s** to come up before the benchmark started, which is itself
  a meaningful operational penalty for this GGUF path.
- **HTTP 400s are likely slot-context overflows, not model instability.** At `-c 32768 --parallel 32`,
  llama-server effectively gives each slot about **1024 tokens** of context budget; longer ShareGPT
  turns can exceed that and fail early, similar to the smaller-model llama.cpp runs at tighter per-slot
  budgets.
