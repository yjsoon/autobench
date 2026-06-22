---
title: SmolLM3-3B · llama.cpp · Q4_K_M
model: HuggingFaceTB/SmolLM3-3B
company: Hugging Face
family: SmolLM
params: 3B
engine: llama.cpp
quant: Q4_K_M
quant_rationale: Q4_K_M — best size/speed/quality balance and the default 4-bit for the shakedown.
source_repo: ggml-org/SmolLM3-3B-GGUF
download_url: https://huggingface.co/ggml-org/SmolLM3-3B-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [smollm3-3b, Hugging Face, SmolLM, Q4_K_M, ≤4B]
status: done
prefill_toks: 1044.80
decode_toks: 653.55
mem_gb: 16.19
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-21
completed_at: 2026-06-21 21:12 +08
run_command: |
  # llama-server (NGC dispatcher image) + ShareGPT serving benchmark, conc=32
  docker run --rm --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/SmolLM3-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 32 -cb \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model SmolLM3-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

Shakedown configuration — re-run under the **ShareGPT serving-benchmark** methodology
(the synthetic `pp512`/`tg128` llama-bench numbers from the first smoke test are superseded).

- **Workload:** real ShareGPT V3 prompts (first human turn), 1000-entry / 15-min cap.
  Completed **974/1000** in **375 s** at **concurrency 32**; 26 prompts (~2.6%) exceeded the
  2048-tok-per-slot context (`-c 65536` ÷ `--parallel 32`) and returned HTTP 400 — an artifact
  of the slot split, not a model failure. Long-prompt coverage would need a higher `-c`.
- **Throughput (aggregate, under load):** prefill **1044.8 tok/s** (input), decode **653.6 tok/s**
  (output). Per-stream: TTFT median **321 ms**, TPOT median **41.6 ms** (≈24 tok/s/stream).
  These are *system* throughput at conc 32 — not comparable to the old single-stream llama-bench
  peaks; the serving numbers are what matters for the deployment question.
- **Memory: 16.19 GB** — up sharply from the 2.94 GB weights-only smoke test. The jump is the
  **64k-token KV cache across 32 slots**, which now dominates footprint. This is the headline
  cost of running a real concurrent serving load, and is the number to compare across models.
- GGUF: `ggml-org/SmolLM3-3B-GGUF` → `SmolLM3-Q4_K_M.gguf` (1.78 GiB weights). Backend: CUDA on
  the GB10, `-ngl 99`, continuous batching (`-cb`). llama.cpp build `b9744-063d9c156`.
- SmolLM3 is a **reasoning model** — it streams `reasoning_content` before `content`; the harness
  counts the first reasoning token as TTFT (else TTFT looks like the full generation time).
