---
title: DeepSeek-R1-0528-Qwen3-8B · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-R1-0528-Qwen3-8B
company: DeepSeek
family: DeepSeek
params: 8B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from unsloth (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF
download_url: https://huggingface.co/unsloth/DeepSeek-R1-0528-Qwen3-8B-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [deepseek-r1-0528-qwen3-8b, DeepSeek, Q4_K_M, 5-15B, conc-32]
status: done
prefill_toks: 256.08
decode_toks: 374.69
mem_gb: 24.22
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 04:10 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model DeepSeek-R1-0528-Qwen3-8B-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A reasoning distill on a Qwen3-8B base — the fastest-decoding dense 8B in the set.** DeepSeek's
R1-0528 reasoning trace distilled onto Qwen3-8B, Q4_K_M from unsloth.

- **Workload:** ShareGPT V3, concurrency 32. **982/1000, 18 errors** (slot-split) in **667 s** —
  **did not hit the time cap**. Loaded in ~17 s.
- **Throughput (aggregate, conc 32):** prefill **256.1 tok/s**, decode **374.7 tok/s** — edging past
  Llama-3.1-8B (365) and clear of the Granite-4.1-8B hybrid (324) on the identical engine/quant/ctx.
  TTFT median **745 ms**, TPOT median **70.7 ms** (≈14 tok/s/stream), req throughput 1.47/s.
- **It's a reasoning model, and it shows in the token mix.** 249.7 k completion tokens over 982
  requests ≈ **254 each** — essentially every request ran to the 256 max-tokens cap (the model keeps
  thinking), vs ~215 each for Llama. That long-output behaviour, plus the heavier prefill batching,
  drives the **high TTFT (745 ms)** — first tokens wait behind fuller decode batches — even though
  steady-state per-token decode (70.7 ms TPOT) is the best of the dense 8Bs. Prefill tok/s (256) reads
  low for the same reason: the run is decode-dominated, so less of the wall-clock is spent on prefill.
- **Memory:** **24.2 GB** — between Llama-3.1-8B (22.7) and Granite-8B (25.4), as expected for a dense
  Qwen3-8B with the standard full-64K transformer KV pre-allocated.
- **Slot-split errors (18):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
