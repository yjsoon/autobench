---
title: Granite 4.1 8B · llama.cpp · Q4_K_M
model: ibm-granite/granite-4.1-8b
company: IBM
family: Granite
params: 8B (hybrid Mamba-2)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from IBM's own GGUF repo (ibm-granite/granite-4.1-8b-GGUF) — official, widest llama.cpp coverage.
source_repo: ibm-granite/granite-4.1-8b-GGUF
download_url: https://huggingface.co/ibm-granite/granite-4.1-8b-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [IBM, Granite, Q4_K_M, 5-15B]

status: done
prefill_toks: 271.57
decode_toks: 323.81
mem_gb: 25.41
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 03:48 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/granite-4.1-8b-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model granite-4.1-8b-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The 8B sibling of the record-setting Granite-4.1-3B — same hybrid Mamba-2 architecture, scaled up.**
IBM's Granite 4.1 8B, Q4_K_M from IBM's own GGUF repo.

- **Workload:** ShareGPT V3, concurrency 32. **985/1000, 15 errors** (slot-split) in **658 s** —
  **did not hit the time cap**. Loaded in **18 s**.
- **Throughput (aggregate, conc 32):** prefill **271.6 tok/s**, decode **323.8 tok/s**. TTFT median
  **290 ms**, TPOT median **85.3 ms** (≈11.7 tok/s/stream), req throughput 1.50/s.
- **Decode scales down with size as expected** — Granite-4.1-3B (hybrid) hit 617; this 8B, ~2.7× the
  parameters, lands at **324**, with no concurrency cliff. Memory **25.4 GB** at 64K context.
- **But the hybrid did NOT win the 8B head-to-head — the dense Llama-3.1-8B beat it on both axes.**
  On the identical engine/quant/ctx, Llama-3.1-8B decoded **365 tok/s at 22.66 GB** vs this model's
  **324 tok/s at 25.41 GB**. The Mamba-2 hybrid's cheap-KV advantage only pays off at *long* per-slot
  context; under this `-c 65536 --parallel 32` split, each slot holds just **2048 tokens**, so there's
  almost no KV to save and the SSM scan's own overhead dominates. The hybrid's edge is real but
  **context-length dependent** — it led at 3B but lost to a well-optimized dense kernel at 8B under
  this short-per-slot 32-way workload. See the Llama-3.1-8B page for the full comparison.
- **Slot-split errors (15):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
