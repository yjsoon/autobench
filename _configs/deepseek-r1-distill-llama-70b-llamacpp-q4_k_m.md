---
title: DeepSeek-R1-Distill-Llama-70B · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-R1-Distill-Llama-70B
company: DeepSeek
family: DeepSeek
params: 70B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from unsloth (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF
download_url: https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [deepseek-r1-distill-llama-70b, DeepSeek, Q4_K_M, 41-130B]
status: done
prefill_toks: 53.57
decode_toks: 52.82
mem_gb: 74.03
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 16:30 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model DeepSeek-R1-Distill-Llama-70B-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The dense-70B reasoning sibling — same Llama-70B speed tier as Llama-3.3-70B, with reasoning outputs
cutting completions further.** DeepSeek's R1 distilled onto Llama-3-70B, unsloth Q4_K_M.

- **Workload:** ShareGPT V3, concurrency 32. **233/1000, 6 errors**, **hit the 15-min cap** (drained at
  1104 s).
- **Throughput (aggregate, conc 32):** prefill **53.6 tok/s**, decode **52.8 tok/s**. TTFT median
  **6.5 s**, TPOT median **473 ms** (≈2.1 tok/s/stream), req throughput 0.21/s.
- **Same 70B bandwidth wall as the base Llama, plus a reasoning tax.** Decode **52.8** sits right with
  Llama-3.3-70B's **48.7** — identical dense-70B architecture, so the same ~273 GB/s memory bandwidth
  caps per-token decode regardless of the fine-tune. On top of that it's a reasoning model emitting long
  traces (58 k completion tokens over 233 reqs ≈ 250 each, at the 256 cap), so it completed even fewer
  prompts before the wall (233 vs 273) and shows a much higher TTFT (6.5 s — first tokens wait behind
  long in-flight generations).
- **Memory: 74.0 GB** — ~40 GB Q4_K_M weights + Llama GQA KV at 64K ctx, a touch above the base
  Llama-3.3 (70.9 GB) from run-to-run KV variation.
- **Slot-split errors (6):** `-c 65536 --parallel 32` → 2048 tok/slot. Low only because few requests
  finished.
