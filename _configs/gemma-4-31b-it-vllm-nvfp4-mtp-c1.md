---
title: Gemma 4 31B · vLLM · NVFP4 + MTP · conc 1
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 31B (dense)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + Google's official MTP assistant drafter (google/gemma-4-31B-it-assistant) via vLLM's native gemma-4 MTP path. conc-1 point of the MTP concurrency sweep (c1 / c8 / c32) on the heaviest Gemma-4 dense model — single-stream latency + peak MTP acceptance.
source_repo: nvidia/Gemma-4-31B-IT-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 1
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B, conc-1]
status: done
prefill_toks: 3.87
decode_toks: 18.31
mem_gb: 107.10
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + Gemma4 MTP head
spec_acceptance: mean acceptance length ~3.06 (2.61–3.61) · avg draft acceptance ~69% (54–87%) · per-position ~0.82/0.68/0.57 (num_speculative_tokens=3)
measured_on: 2026-07-02
completed_at: 2026-07-02 15:52 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:e414712fdc04f61d98ccc58cb61232a0587a8c024544e9e6cf12f97b19b38172
run_command: |
  # nightly-aarch64 (same digest as the conc-32 MTP row). NVFP4 base + Google's native gemma-4 MTP
  # assistant drafter. Bench sidecar hits the engine on :8000 directly (500 prompts, 300 s cap).
  VLLM_IMAGE=vllm/vllm-openai:nightly-aarch64 scripts/bench-vllm-serving.sh nvidia/Gemma-4-31B-IT-NVFP4 65536 1 500 300 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3}'
  # 23/500 prompts (300 s time cap), 0 errors. TTFT median 327.7 ms, TPOT median 52.6 ms, req thr 0.074/s.
  # Ready after 471 s (warm compile cache). SpecDecoding: mean accept-len ~3.06, avg draft acceptance ~69%.
---

**Conc-1 point of the Gemma 4 31B MTP sweep — single-stream latency, and the *best* MTP acceptance of the
sweep (~3.06).** NVIDIA NVFP4 base + Google's official `google/gemma-4-31B-it-assistant` MTP drafter, on
the maintained vLLM `nightly-aarch64`, ctx 65536, conc 1.

- **Load:** ready in **471 s** (NVFP4 weights + MTP head + CUDA-graph capture; warm torch.compile cache).
- **Workload:** ShareGPT V3, concurrency 1. **23/500 completed, 0 errors** before the **300 s time cap**
  (few prompts finish single-stream in a fixed window — this is a latency/acceptance point, not a
  throughput point).
- **Throughput / latency:** decode **18.31 tok/s** (single stream), prefill **3.87 tok/s** (aggregate
  prefill is low here only because one stream over short ShareGPT prompts barely exercises prefill — not
  a regression). TTFT median **327.7 ms**, TPOT median **52.6 ms** — the lowest per-token latency of the
  sweep. Peak mem **107.10 GB** (vLLM static KV reservation at util 0.85 + MTP head), not the footprint.
- **MTP acceptance — peaks here at ~3.06.** Across the run: **mean acceptance length ~3.06 (2.61–3.61)**,
  **avg draft acceptance ~69% (54–87%)**, per-position **~0.82 / 0.68 / 0.57** at
  `num_speculative_tokens=3`. Single-stream has no batching interference on the MTP head, so acceptance
  runs highest — the 3rd draft position lands ~0.57 (vs ~0.44 at c8), lifting mean accept-len past 3.0.
- **Concurrency sweep (same model/engine/quant/spec, ctx 65536):**

  | Conc | decode (agg) | TTFT med | TPOT med | mean accept-len | avg draft accept |
  |---|--:|--:|--:|--:|--:|
  | **1**  | **18.31**  | **327.7 ms** | **52.6 ms** | **~3.06** | **~69%** |
  | 8  | 117.05 | 536.4 ms | 61.2 ms | ~2.85 | ~61% |
  | 32 | 323.5  | 685.1 ms | 86.3 ms | ~2.80 | ~58–65% |

  Draft acceptance is highest at conc-1 (~3.06) and drifts down as concurrency rises (batching
  interference on the single MTP head), while aggregate decode scales the other way (18→117→324).
- Text path benchmarked (`mm_served: false`).
