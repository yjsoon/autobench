---
title: Qwen3.6-27B · vLLM · FP8 + MTP · conc 8
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
concurrency: 8
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B, conc-8]
status: done
prefill_toks: 115.13
decode_toks: 97.96
mem_gb: 103.71
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 70% avg draft acceptance · mean acceptance length 3.1 · per-position 0.86/0.69/0.53
measured_on: 2026-06-22
completed_at: 2026-06-22 19:42 +08
run_command: |
  # base Qwen3.6-27B-FP8 + native MTP (mtp.safetensors ships in-repo) via vLLM --speculative-config.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-27B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256

# SpecDecoding metrics (vLLM, steady-state): Mean acceptance length ~3.1, Avg draft acceptance ~70%,
# per-position acceptance 0.86 / 0.69 / 0.53 (num_speculative_tokens=3).
---

**conc-8 point of the Qwen3.6-27B MTP sweep — acceptance holds, latency is the story.** Qwen3.6-27B
FP8 + native MTP at **concurrency 8** (500-prompt / 300 s latency-characterization cap).

- **Workload:** ShareGPT V3, concurrency 8. **118/500, 0 errors** (hit the 300 s cap — expected at low
  concurrency, where 8 streams take longer to clear 500 prompts).
- **Per-stream latency:** TPOT median **66.3 ms** (≈15 tok/s/stream), TTFT median **624 ms**.
- **Acceptance: ~70% avg draft acceptance, mean length ~3.1** — *holds* vs the conc-32 run's 67%
  (per-position 0.86 / 0.69 / 0.53). This is the clean, comparable result: **acceptance is
  workload-driven and barely moves with concurrency**, exactly as the published MTP behaviour predicts.
- **Reading the aggregate tok/s:** decode **98** / prefill **115** are *lower* than the conc-32 run
  (241 / 242) — **not a regression**, just concurrency scaling: 8 concurrent streams push far fewer
  total tokens/s than 32. The meaningful low-concurrency metric is the per-stream **TPOT** above, not
  aggregate throughput. The MTP *speedup* at conc 8 (vs no-MTP at conc 8) would need a matched
  base-at-conc-8 run, which isn't in this spec-only sweep — see the conc-1 sibling for the
  single-stream latency point.

**Queued — concurrency-8 variant of [Qwen3.6-27B · vLLM · FP8 + MTP].** Low-concurrency point of the Qwen3.6 native-MTP sweep (cap 500 prompts / 300 s, latency characterization). Compare decode + acceptance against the conc-32 run to see how the MTP gain scales as the batch empties.
