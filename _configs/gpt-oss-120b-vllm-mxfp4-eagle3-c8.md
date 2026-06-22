---
title: gpt-oss-120b · vLLM · MXFP4 + EAGLE3 · conc 8
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: vLLM
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss MXFP4 base + NVIDIA's throughput-tuned EAGLE3 draft head (nvidia/gpt-oss-120b-Eagle3-throughput) — the spec-decode dimension on the gpt-oss-120b headliner. gpt-oss is text-only, so it should dodge vLLM's multimodal draft-model block that complicated Gemma.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
concurrency: 8
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-8]
status: done
prefill_toks: 80.16
decode_toks: 50.02
mem_gb: 107.00
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-22
completed_at: 2026-06-22 23:44 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, NVIDIA EAGLE3-throughput draft.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"model":"nvidia/gpt-oss-120b-Eagle3-throughput","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**The acceptance floor of the whole sweep — at conc-8 the 120b EAGLE3 draft acceptance is essentially
zero.** gpt-oss-120b MXFP4 + NVIDIA EAGLE3-throughput draft, vLLM, conc 8.

- **Load:** **543 s** (~9 min) to ready.
- **Workload:** ShareGPT V3, concurrency 8. **65/500 completed, 15 errors** (the most of the sweep) before
  the **300 s time cap** (`hit_time_cap=true`).
- **Throughput:** prefill **80.16 tok/s**, decode **50.02 tok/s** aggregate (~6.3 tok/s/stream).
- **EAGLE3 acceptance: collapsed to ~zero.** mean acceptance length **~1.01–1.10 (centered ~1.05)**, avg
  draft acceptance **~0.3–3.3% (centered ~1.5%)**, per-position **~0.02–0.07 / ~0 / ~0**. The model drafts
  **~180–200 tok/s but accepts ~3** — past the first slot the draft is **never** right. This is the lowest
  acceptance measured anywhere in the benchmark.
- **Confirms the concurrency-degradation trend across the full sweep:**

  | gpt-oss EAGLE3 | conc | mean accept-len | avg draft accept | decode tok/s |
  |---|---|---|---|---|
  | 20b | 1 | ~1.7 | ~25% | 38.6 /stream |
  | 20b | 8 | ~1.2 | ~6% | 126.3 agg |
  | 120b | 1 | ~1.25 | ~9% | 14.7 /stream |
  | **120b** | **8** | **~1.05** | **~1.5%** | **50.0 agg** |

  Acceptance falls monotonically as concurrency rises and as the model grows — the **opposite** of the
  "acceptance is workload-driven, ~constant across concurrency" rule of thumb (CLAUDE.md). The draft heads
  (RedHatAI for 20b, NVIDIA-throughput for 120b) just don't predict the ShareGPT + harmony-reasoning token
  stream, and batching makes verification even less effective.
- **Harmony corruption (15 errors):** same garbled-draft parser failures, worst at this highest-concurrency
  120b point — e.g. `unexpected tokens remaining in message header: Some("scientific progress!!! societal!!! discovery!!!!!")`
  and repeated `Unexpected token 0 while expecting start token 200006`.
- **Harmony artifacts:** TTFT median **30.2 s** (buffered reasoning + 8-way queue on a slow 120b) / TPOT
  **0.0** — not real latencies.
- **Memory: 107.0 GB** = vLLM 0.85 reservation + EAGLE3 head.
- **Sweep conclusion:** the published [conc-32] result (20b +28% / 120b −45%) captured only one slice. Across
  conc 1/8 the EAGLE3 drafts for gpt-oss are a **poor fit on ShareGPT** — low and concurrency-degrading
  acceptance, plus intermittent harmony-stream corruption. Where spec-decode helped (20b conc-32) it was a
  scheduling/prefill effect, not high draft acceptance. For real gain on these models, prefer EAGLE3 on
  **coding** workloads (where acceptance is higher) or skip spec-decode for general chat.
