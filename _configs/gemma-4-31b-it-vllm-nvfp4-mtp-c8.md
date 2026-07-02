---
title: Gemma 4 31B · vLLM · NVFP4 + MTP · conc 8
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 31B (dense)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + Google's official MTP assistant drafter (google/gemma-4-31B-it-assistant) via vLLM's native gemma-4 MTP path. conc-8 point of the MTP concurrency sweep (c1 / c8 / c32) on the heaviest Gemma-4 dense model.
source_repo: nvidia/Gemma-4-31B-IT-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 8
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B, conc-8]
status: done
prefill_toks: 137.27
decode_toks: 117.05
mem_gb: 107.09
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + Gemma4 MTP head
spec_acceptance: mean acceptance length ~2.85 (2.50–3.21) · avg draft acceptance ~61% (50–73.5%) · per-position ~0.78/0.60/0.44 (num_speculative_tokens=3)
measured_on: 2026-07-02
completed_at: 2026-07-02 15:38 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:e414712fdc04f61d98ccc58cb61232a0587a8c024544e9e6cf12f97b19b38172
run_command: |
  # nightly-aarch64 (same digest as the conc-32 MTP row). NVFP4 base + Google's native gemma-4 MTP
  # assistant drafter. Bench sidecar hits the engine on :8000 directly (500 prompts, 300 s cap).
  VLLM_IMAGE=vllm/vllm-openai:nightly-aarch64 scripts/bench-vllm-serving.sh nvidia/Gemma-4-31B-IT-NVFP4 65536 8 500 300 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3}'
  # 162/500 prompts (300 s time cap), 0 errors. TTFT median 536.4 ms, TPOT median 61.2 ms, req thr 0.516/s.
  # Ready after 489 s (warm compile cache). SpecDecoding: mean accept-len ~2.85, avg draft acceptance ~61%.
---

**Conc-8 point of the Gemma 4 31B MTP sweep — acceptance holds at ~2.85, mid-way between the conc-1 and
conc-32 rows.** NVIDIA NVFP4 base + Google's official `google/gemma-4-31B-it-assistant` MTP drafter, on
the maintained vLLM `nightly-aarch64`, ctx 65536, conc 8.

- **Load:** ready in **489 s** (NVFP4 weights + MTP head + CUDA-graph capture; warm torch.compile cache).
- **Workload:** ShareGPT V3, concurrency 8. **162/500 completed, 0 errors** before the **300 s time cap**.
- **Throughput:** decode **117.05 tok/s** aggregate, prefill **137.27 tok/s**. TTFT median **536.4 ms**,
  TPOT median **61.2 ms**, req throughput **0.516/s**. Peak mem **107.09 GB** (vLLM static KV reservation
  at util 0.85 + MTP head), not the footprint.
- **MTP acceptance — steady at ~2.85.** Across the run: **mean acceptance length ~2.85 (2.50–3.21)**,
  **avg draft acceptance ~61% (50–73.5%)**, per-position **~0.78 / 0.60 / 0.44** at
  `num_speculative_tokens=3` — the draft head reliably lands token 1 (~0.78) with the 2nd/3rd decaying as
  expected, and comfortably clear of the near-1.0 that would flag a target/draft mismatch.
- **Concurrency sweep (same model/engine/quant/spec, ctx 65536):**

  | Conc | decode (agg) | TTFT med | TPOT med | mean accept-len | avg draft accept |
  |---|--:|--:|--:|--:|--:|
  | 1  | 18.31  | 327.7 ms | 52.6 ms | ~3.06 | ~69% |
  | **8**  | **117.05** | **536.4 ms** | **61.2 ms** | **~2.85** | **~61%** |
  | 32 | 323.5  | 685.1 ms | 86.3 ms | ~2.80 | ~58–65% |

  Aggregate decode scales with concurrency (18→117→324) while per-token latency (TPOT) climbs with
  contention (52.6→61.2→86.3 ms) — the standard throughput-vs-latency curve. Draft acceptance drifts
  *down* slightly as concurrency rises (batching interference on the single MTP head), still ≥2.8 at c32.
- Text path benchmarked (`mm_served: false`).
