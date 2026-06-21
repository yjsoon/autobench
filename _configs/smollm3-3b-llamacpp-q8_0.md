---
title: SmolLM3-3B · llama.cpp · Q8_0
model: HuggingFaceTB/SmolLM3-3B
company: Hugging Face
family: SmolLM
params: 3B
engine: llama.cpp
quant: Q8_0
quant_rationale: Q8_0 — near-lossless reference point to quantify Q4_K_M's quality/speed tradeoff.
source_repo: ggml-org/SmolLM3-3B-GGUF
download_url: https://huggingface.co/ggml-org/SmolLM3-3B-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [Hugging Face, SmolLM, Q8_0, ≤4B]

status: done
prefill_toks: 914.23
decode_toks: 570.60
mem_gb: 17.67
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-21
completed_at: 2026-06-21 21:20 +08
run_command: |
  # llama-server (NGC dispatcher image) + ShareGPT serving benchmark, conc=32
  docker run --rm --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/SmolLM3-Q8_0.gguf -ngl 99 -c 65536 --parallel 32 -cb \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model SmolLM3-Q8_0.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

Q8_0 companion to the Q4_K_M shakedown — the quant axis, re-run under the **ShareGPT
serving-benchmark** methodology (synthetic llama-bench numbers superseded).

- **Workload:** real ShareGPT V3 prompts, 1000-entry / 15-min cap. Completed **974/1000** in
  **428 s** at **concurrency 32** (same 26 long prompts exceeded the 2048-tok/slot context).
- **Throughput (aggregate, under load):** prefill **914.2 tok/s**, decode **570.6 tok/s**.
  Per-stream: TTFT median **342 ms**, TPOT median **49.1 ms**.
- **vs Q4_K_M** (same load): decode **570.6 vs 653.6 tok/s** — Q8 moves ~2× the bytes per
  weight and decode is bandwidth-bound, so it's slower; the gap (≈13%) is much smaller than the
  single-stream llama-bench gap (70.6 vs 105.7) because at conc 32 the KV-cache traffic, not the
  weights, dominates bandwidth. Memory **17.67 vs 16.19 GB** (≈1.3 GB heavier weights; the 64k KV
  cache is the bulk of both).
- GGUF: `ggml-org/SmolLM3-3B-GGUF` → `SmolLM3-Q8_0.gguf` (3.04 GiB weights). CUDA on the GB10,
  `-ngl 99`, continuous batching. llama.cpp build `b9744-063d9c156`.
