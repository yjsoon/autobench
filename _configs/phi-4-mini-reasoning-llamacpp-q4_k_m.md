---
title: Phi-4-mini-reasoning · llama.cpp · Q4_K_M
model: microsoft/Phi-4-mini-reasoning
company: Microsoft
family: Phi
params: 3.8B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer — Microsoft publishes no GGUF). Widest llama.cpp coverage, strong size/quality balance.
source_repo: bartowski/microsoft_Phi-4-mini-reasoning-GGUF
download_url: https://huggingface.co/bartowski/microsoft_Phi-4-mini-reasoning-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [phi-4-mini-reasoning, Microsoft, Phi, Q4_K_M, ≤4B, conc-32]
status: done
prefill_toks: 402.97
decode_toks: 552.37
mem_gb: 21.18
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 03:24 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model microsoft_Phi-4-mini-reasoning-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A fast dense 3.8B — and half of the pair that debunked the "llama.cpp can't do concurrency"
theory.** Microsoft's Phi-4-mini-reasoning, Q4_K_M from bartowski (Microsoft ships no GGUF).

- **Workload:** ShareGPT V3, concurrency 32. **982/1000, 18 errors** (slot-split) in **452 s** —
  **no time-cap**. Loaded in 16 s.
- **Throughput (aggregate, conc 32):** prefill **403.0 tok/s**, decode **552.4 tok/s**. TTFT median
  **375 ms**, TPOT median **49.8 ms** (≈20 tok/s/stream). Second only to the Granite-4.1-3B Mamba
  hybrid (617) among the small models — and **3.9× faster decode than the dense Nemotron-Nano-4B**
  (141) on the *identical* engine/quant/ctx, despite being about the same size. That contrast is the
  evidence that llama.cpp parallelizes 32-way serving fine on GB10; the Nano-4B's slowness is
  model-specific, not an engine ceiling.
- **A dense transformer can't quite match the Mamba hybrid.** Phi (dense, 552) vs Granite-4.1-3B
  (hybrid Mamba-2, 617) at the same size/engine: the hybrid's cheaper KV/decode still wins, but a
  well-behaved dense model is close. Memory **21.2 GB** here (full 64K transformer KV pre-allocated)
  vs Granite's 16.9 GB reflects exactly that KV difference.
- **Slot-split errors (18):** `-c 65536 --parallel 32` → 2048 tok/slot; longer prompts 400. Standard
  llama.cpp-run artifact.
