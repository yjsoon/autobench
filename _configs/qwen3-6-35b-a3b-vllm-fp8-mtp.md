---
title: Qwen3.6-35B-A3B · vLLM · FP8 + MTP
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
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B, conc-32]
status: done
prefill_toks: 420.12
decode_toks: 407.93
mem_gb: 106.55
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 67% avg draft acceptance · mean acceptance length 3.0 · per-position 0.84/0.66/0.51
measured_on: 2026-06-22
completed_at: 2026-06-22 19:28 +08
run_command: |
  # planned: base Qwen3.6-35B-A3B-FP8 + native MTP via vLLM --speculative-config (method finalized at
  # run time; qwen3.6 ships mtp.safetensors in-repo).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**Native MTP lifts the Qwen3.6 MoE +43% — a clean win, and acceptance matches the dense 27B.**
Qwen3.6-35B-A3B FP8 + the model's built-in MTP head, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **627 s** — clean, no time cap.
- **Throughput (aggregate, conc 32):** prefill **420.1 tok/s**, decode **407.9 tok/s** vs the base
  Qwen3.6-35B-A3B's **286.0** → **+43% decode** from MTP.
- **Acceptance rate (captured from vLLM `SpecDecoding metrics`):** **mean acceptance length ≈ 3.0**,
  **avg draft acceptance ≈ 67%**, per-position **0.84 / 0.66 / 0.51** — essentially identical to the
  dense Qwen3.6-27B MTP (67%), which is expected: **acceptance is workload-driven** (this ShareGPT chat
  load), not model-size-driven. It sits a bit under the ~80%+ Qwen reports on *coding* workloads —
  general chat is less predictable per token. (Cross-checked against published MTP rates per CLAUDE.md.)
- **A smaller % gain than the dense 27B (+43% vs +56%) — and that's the headroom story again.** The MoE
  base is already fast (286 vs the dense 27B's 155) because it activates only ~3B params, so there's
  *less* idle compute at conc 32 for the MTP draft to convert into throughput. Same ~67% acceptance,
  smaller relative lift — the faster the base, the less spec-decode adds at a given concurrency. Both
  Qwen3.6 models still win clearly at conc 32, unlike the saturated 120B gpt-oss.
- **Memory: 106.6 GB is the vLLM `--gpu-memory-utilization 0.85` reservation** (+ MTP head), not the
  footprint (FP8 weights ≈ 35 GB).
