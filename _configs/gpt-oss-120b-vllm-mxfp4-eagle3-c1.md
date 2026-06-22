---
title: gpt-oss-120b · vLLM · MXFP4 + EAGLE3 · conc 1
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
concurrency: 1
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-1]
status: done
prefill_toks: 7.22
decode_toks: 14.69
mem_gb: 107.02
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-22
completed_at: 2026-06-22 23:29 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, NVIDIA EAGLE3-throughput draft.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"model":"nvidia/gpt-oss-120b-Eagle3-throughput","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Single-stream point for the 120b EAGLE3 sweep — the draft is near-dead weight even with maximum
headroom.** gpt-oss-120b MXFP4 + NVIDIA's throughput-tuned EAGLE3 draft (`nvidia/gpt-oss-120b-Eagle3-throughput`),
vLLM, conc 1.

- **Load:** **589 s** (~10 min) to ready — the 120b weight load + EAGLE3 head + CUDA-graph capture.
- **Workload:** ShareGPT V3, concurrency 1. **20/500 completed, 3 errors** before the **300 s time cap**
  (`hit_time_cap=true`) — at conc-1 the 120b is slow enough that only ~20 requests finish in 5 min.
- **Throughput:** decode **14.69 tok/s** (single stream), prefill **7.22 tok/s** (the prefill figure is
  depressed by the tiny completed count + harmony buffering, not a true compute rate).
- **EAGLE3 acceptance: ~dead.** mean acceptance length **~1.05–1.44 (centered ~1.25)**, avg draft acceptance
  **~1.3–14.7% (centered ~9%)**, per-position **~0.05–0.36 / 0–0.07 / ~0**. With `num_speculative_tokens=3`
  the model drafts ~45 tok/s but **accepts only ~4** — the second/third draft positions almost never hit.
  Even at conc-1 (maximum headroom) the NVIDIA throughput-EAGLE3 head simply doesn't predict this workload
  (ShareGPT chat + harmony reasoning), so spec-decode is pure overhead.
- **Harmony corruption (3 errors):** garbled draft output again broke the parser — e.g.
  `unexpected tokens remaining in message header: Some("!! (? ) analysis![]!????")` and
  `"!assistant The!! You!analysis:"`. Same EAGLE3 × harmony interaction seen on the 20b.
- **Harmony artifacts:** TTFT median **12.2 s** / TPOT **0.0** are buffered-reasoning artifacts; aggregate
  tok/s are the valid headline.
- **Memory: 107.0 GB** = vLLM 0.85 reservation + EAGLE3 head, not footprint.
- **Consistent with the [conc-32] −45% finding:** the 120b gains nothing from EAGLE3 at any concurrency —
  at conc-32 it's compute-saturated, and at conc-1 the draft acceptance is too low to help. There is no
  concurrency regime where this EAGLE3 head pays off on this workload.
