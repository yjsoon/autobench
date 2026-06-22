---
title: gpt-oss-120b · vLLM · MXFP4
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: vLLM
quant: MXFP4
quant_rationale: gpt-oss's native MXFP4 (~63 GB); FP4-accelerated on Blackwell with the CUTLASS kernels.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-32]
status: done
prefill_toks: 278.76
decode_toks: 252.81
mem_gb: 107.91
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 15:30 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly. Harmony vocab pre-seeded (-v ~/models/tiktoken_cache:/vocab:ro,
  # TIKTOKEN_ENCODINGS_BASE=/vocab) — see CLAUDE.md.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The vLLM half of the gpt-oss-120b engine comparison — markedly higher raw throughput than SGLang,
but the same harmony-parse robustness cost, amplified at 120B.** OpenAI's gpt-oss-120b (116.8B total /
5.1B active MoE, native MXFP4) on the cu130-nightly vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **788/1000 completed, 212 errors**, in **776 s** (no time
  cap). Model loaded + CUDA-graph captured in **482 s** (the 122 GB on-disk checkpoint was already
  cached).
- **Aggregate throughput (conc 32):** prefill **278.8 tok/s**, decode **252.8 tok/s** — total tokens ÷
  wall-clock, valid over the 788 successful requests. **vs the SGLang run of the same model
  (187.7 / 140.3): vLLM is ~1.5× prefill and ~1.8× decode** — the same engine ordering the 20b
  comparison showed, and a big jump over the ~56–76 tok/s single-stream figures reported elsewhere for
  a Spark (this is 32-way aggregate).
- **The harmony caveats are the same as gpt-oss-20b-vLLM, worse here:**
  - **212 errors (~27%)** — harmony-parser faults (`Unexpected token … expecting start token 200006`,
    `Unknown role …`). With `--max-tokens 256`, the heavily-reasoning 120b is truncated mid-thought even
    more often than the 20b (which logged ~11%), leaving incomplete harmony structures that vLLM's
    `stream_harmony` finalizer rejects. SGLang's `--reasoning-parser gpt-oss` tolerated the same
    truncation cleanly — a real engine robustness gap, widening with model reasoning depth.
  - **TTFT median 29.6 s / TPOT 0.0** are **not meaningful** — vLLM's harmony chat path buffers the
    reasoning channel and emits the final message in one burst, so our per-token client metrics
    collapse. The aggregate tok/s above are unaffected.
- **Memory 107.9 GB is a reservation, not a footprint** (`--gpu-memory-utilization 0.85` of the 121 GB
  pool); the MXFP4 weights are ~63 GB. Comparable to the SGLang run's reservation.
- **Takeaway for the engine choice:** for raw 32-way throughput vLLM wins clearly, but for gpt-oss
  *as-served via Chat Completions with a tight token budget*, SGLang's gpt-oss parser is the robust
  path (zero errors, clean latency). Pick by whether you need peak throughput or correct streaming.
