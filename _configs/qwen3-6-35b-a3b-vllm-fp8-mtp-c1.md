---
title: Qwen3.6-35B-A3B · vLLM · FP8 + MTP · conc 1
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
concurrency: 1
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B, conc-1]
status: done
prefill_toks: 89.67
decode_toks: 53.97
mem_gb: 106.24
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 66% avg draft acceptance · mean acceptance length 3.0 · per-position 0.87/0.67/0.54
measured_on: 2026-06-22
completed_at: 2026-06-22 20:24 +08
run_command: |
  # planned: base Qwen3.6-35B-A3B-FP8 + native MTP via vLLM --speculative-config (method finalized at
  # run time; qwen3.6 ships mtp.safetensors in-repo).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**Single-stream point — ~65 tok/s/stream latency, and the 35B-A3B MTP sweep closes with acceptance
flat.** The sparse MoE (3B active) + native MTP at **concurrency 1** (300 s cap → 64 prompts).

- **Per-stream latency:** TPOT median **15.5 ms** (≈**65 tok/s single-stream**) — the lowest of the
  whole MTP sweep, and ~3.8× faster per-token than the dense Qwen3.6-27B MTP at conc-1 (58 ms): the MoE
  activates only ~3B params, so single-stream decode is very cheap. TTFT median 148 ms.
- **Acceptance: ~66%, mean length 3.0** (per-position 0.87 / 0.67 / 0.54).
- **The 35B-A3B MTP concurrency sweep — acceptance holds across a 32× range:**

  | concurrency | avg draft acceptance | mean accept length | TPOT median |
  |---|---|---|---|
  | 32 | 67% | 3.0 | — |
  | 8 | 68% | 3.0 | 35.2 ms |
  | **1** | **66%** | **3.0** | **15.5 ms** |

  Same conclusion as the dense 27B sweep: **MTP draft acceptance is workload-driven (~66–68% on
  ShareGPT here), essentially flat across concurrency** — what changes is per-token latency (TPOT drops
  sharply as the batch empties) and aggregate throughput (scales with streams). The MoE just runs the
  whole curve at much lower latency than the dense model. (MTP *speedup vs no-MTP* at conc 1/8 needs a
  matched base run — deferred.)
