---
title: Qwen3.6-35B-A3B · vLLM · FP8 + MTP · conc 8
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
speculative: MTP
quant: FP8
quant_rationale: Qwen3.6-35B-A3B FP8 + the model's native MTP module (mtp.safetensors ships in-repo) — built-in multi-token-prediction speculative decoding on the sparse MoE.
source_repo: Qwen/Qwen3.6-35B-A3B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 8
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B, conc-8]
status: done
prefill_toks: 260.87
decode_toks: 190.31
mem_gb: 104.30
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 68% avg draft acceptance · mean acceptance length 3.0 · per-position 0.85/0.68/0.53
measured_on: 2026-06-22
completed_at: 2026-06-22 20:11 +08
run_command: |
  # planned: base Qwen3.6-35B-A3B-FP8 + native MTP via vLLM --speculative-config (method finalized at
  # run time; qwen3.6 ships mtp.safetensors in-repo).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**conc-8 point of the Qwen3.6-35B-A3B MTP sweep — fastest per-token latency of any MTP run, acceptance
holds.** The sparse MoE (3B active) + native MTP at **concurrency 8** (500-prompt / 300 s cap).

- **Workload:** ShareGPT V3, concurrency 8. **228/500, 0 errors** (hit the 300 s cap).
- **Per-stream latency:** TPOT median **35.2 ms** (≈28 tok/s/stream) — **about half the dense
  Qwen3.6-27B MTP's 66 ms at the same conc-8**, because the MoE only activates ~3B params/token. TTFT
  median 355 ms.
- **Acceptance: ~68%, mean length 3.0** (per-position 0.85 / 0.68 / 0.53) — holds vs the conc-32 run's
  67%, confirming acceptance is workload-driven (the same ShareGPT MTP head hits ~67–70% at every
  concurrency on both the 27B dense and this 35B MoE).
- **Aggregate decode 190 / prefill 261** are concurrency-scaled (8 streams), lower than the conc-32 run
  (408 / 420) — not a regression; TPOT above is the meaningful low-concurrency metric. The MTP *speedup*
  vs no-MTP at conc 8 needs a matched base-at-conc-8 (deferred).
