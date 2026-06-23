---
title: Qwen3.6-27B · vLLM · NVFP4 + MTP · conc 8
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: Concurrency-8 point of the Qwen3.6-27B NVFP4 + native-MTP sweep — same stack as the conc-32 run (unsloth NVFP4 base + in-repo MTP), lower batch for the latency-characterization point. Acceptance should hold ~constant vs conc-32 (workload-driven); what changes is whether the MTP speedup materializes.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 8
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-8]
status: done
prefill_toks: 120.27
decode_toks: 109.05
mem_gb: 106.65
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 71% avg draft acceptance · mean acceptance length 3.1 · per-position 0.87/0.69/0.55 (num_speculative_tokens=3)
measured_on: 2026-06-23
completed_at: 2026-06-23 10:33 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-8 latency point (500 prompts / 300 s cap, matching the FP8-MTP -c8 convention).
  scripts/bench-vllm-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 8 500 300 256 \
    --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  # 134/500 prompts (hit 300 s cap), 0 errors. TTFT median 579.5 ms, TPOT median 60.8 ms.
---

**conc-8 point of the Qwen3.6-27B NVFP4 + MTP sweep.** Lower-batch latency characterization (cap 500
prompts / 300 s — reached 134 at the time cap, 0 errors).

- **Result (conc 8):** prefill 120.3 / decode **109.05** tok/s aggregate; per-stream TTFT median 579.5 ms,
  TPOT median **60.8 ms** (≈16 tok/s/stream). Peak mem 106.7 GB.
- **Acceptance ~71% / mean accept-len 3.1** (per-position 0.87 / 0.69 / 0.55) — **holds vs the conc-32
  run's 67%** (even a touch higher), confirming acceptance is workload-driven, not concurrency-driven.
  Aggregate tok/s is lower than conc-32 (274) purely because 8 streams push fewer total tokens — the
  meaningful low-conc metric is the per-stream TPOT above.
