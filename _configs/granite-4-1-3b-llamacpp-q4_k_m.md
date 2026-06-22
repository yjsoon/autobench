---
title: Granite 4.1 3B · llama.cpp · Q4_K_M
model: ibm-granite/granite-4.1-3b
company: IBM
family: Granite
params: 3B (hybrid Mamba-2)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from IBM's own GGUF repo (ibm-granite/granite-4.1-3b-GGUF) — official, widest llama.cpp coverage.
source_repo: ibm-granite/granite-4.1-3b-GGUF
download_url: https://huggingface.co/ibm-granite/granite-4.1-3b-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [granite-4.1-3b, IBM, Granite, Q4_K_M, ≤4B, conc-32]
status: done
prefill_toks: 535.36
decode_toks: 617.5
mem_gb: 16.90
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 03:16 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/granite-4.1-3b-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model granite-4.1-3b-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The fastest decode in the entire benchmark so far — and it's the architecture, not the size.**
IBM's Granite 4.1 3B is a **hybrid Mamba-2 / transformer** model, and on the *same* llama.cpp engine
where the dense Nemotron-Nano-4B managed only 141 tok/s decode, this one hits **617.5 tok/s**.

- **Workload:** ShareGPT V3, concurrency 32. **985/1000, 15 errors** (slot-split), in **334 s** —
  **did not hit the time cap** (finished comfortably), unlike every 120B giant and even the dense 4B.
- **Throughput (aggregate, conc 32):** prefill **535.4 tok/s**, decode **617.5 tok/s** — higher than
  the best FP4 MoE on vLLM (Nano-Omni 389) and **4.4× the dense Nemotron-Nano-4B** on the identical
  engine/quant/ctx. TTFT median **144 ms**, TPOT median **44 ms** (≈23 tok/s/stream), req throughput
  2.95/s — all best-in-set.
- **Why:** the **Mamba-2 hybrid** decodes with linear-time state updates and a tiny recurrent state
  instead of a growing quadratic KV cache. That shows up directly in **memory: just 16.9 GB** (vs
  21.6 GB for the smaller-on-paper dense 4B that pre-allocates a full 64K transformer KV) and in the
  flat, fast per-token decode. This is the standout finding of the small-model sweep: at 32-way
  serving on GB10, architecture (hybrid SSM) beats both parameter count and quant format for decode
  throughput. (Note: llama.cpp parallelizes 32-way serving fine here — Phi-4-mini-reasoning, a dense
  3.8B on the same engine, hit 552 tok/s; the Nemotron-Nano-4B's 141 tok/s is a model-specific
  anomaly, not an engine limit. The hybrid just removes the KV cost on top of that.)
- **Slot-split errors (15):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across the llama.cpp runs.
