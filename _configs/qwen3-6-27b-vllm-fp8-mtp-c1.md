---
title: Qwen3.6-27B · vLLM · FP8 + MTP · conc 1
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: FP8
quant_rationale: Qwen3.6-27B FP8 + the model's own native MTP module (mtp.safetensors ships in the base repo) — built-in multi-token-prediction speculative decoding, no separate draft.
source_repo: Qwen/Qwen3.6-27B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B, conc-1]
status: done
prefill_toks: 3.11
decode_toks: 15.15
mem_gb: 102.94
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 68% avg draft acceptance · mean acceptance length 3.0 · per-position 0.89/0.73/0.55
measured_on: 2026-06-22
completed_at: 2026-06-22 19:55 +08
run_command: |
  # base Qwen3.6-27B-FP8 + native MTP (mtp.safetensors ships in-repo) via vLLM --speculative-config.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-27B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256

# SpecDecoding metrics (vLLM, steady-state): Mean acceptance length ~3.0, Avg draft acceptance ~68%,
# per-position acceptance 0.89 / 0.73 / 0.55 (num_speculative_tokens=3).
---

**Single-stream point — lowest per-token latency, and the acceptance cross-check closes cleanly.**
Qwen3.6-27B FP8 + native MTP at **concurrency 1** (latency characterization, 300 s cap → 18 prompts).

- **Per-stream latency:** TPOT median **58.3 ms** (≈17 tok/s/stream) — the **best of the sweep**
  (conc-1 58 ms < conc-8 66 ms), as expected: no batching contention at a single stream, so the MTP
  multi-token emits land with the lowest per-token latency. TTFT median 375 ms.
- **Acceptance: ~68%, mean length 3.0** (per-position 0.89 / 0.73 / 0.55). The first draft position
  hits **0.89** here vs 0.84 at conc-32 — marginally higher single-stream, but essentially flat.
- **The acceptance cross-check, across the full 27B MTP sweep:**

  | concurrency | avg draft acceptance | mean accept length | TPOT median |
  |---|---|---|---|
  | 32 | 67% | 3.0 | — |
  | 8 | 70% | 3.1 | 66.3 ms |
  | **1** | **68%** | **3.0** | **58.3 ms** |

  **Acceptance holds ~67–70% across a 32× concurrency range** — confirming the literature: MTP
  acceptance is **workload-driven (ShareGPT chat), not concurrency-driven**. It sits just under the
  ~80%+ Qwen reports on coding (chat is less predictable). What *does* change with concurrency is
  per-token latency (TPOT improves as the batch shrinks) and aggregate throughput (scales up with more
  streams). **Note:** the MTP *speedup vs no-MTP* at conc 1/8 would require a matched base-at-same-conc
  run, which this spec-only sweep doesn't include.

**Queued — concurrency-1 variant of [Qwen3.6-27B · vLLM · FP8 + MTP].** Low-concurrency point of the Qwen3.6 native-MTP sweep (cap 500 prompts / 300 s, latency characterization). Compare decode + acceptance against the conc-32 run to see how the MTP gain scales as the batch empties.
