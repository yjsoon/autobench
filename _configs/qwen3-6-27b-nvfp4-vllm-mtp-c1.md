---
title: Qwen3.6-27B · vLLM · NVFP4 + MTP · conc 1
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: Single-stream (conc-1) point of the Qwen3.6-27B NVFP4 + native-MTP sweep — the best-case MTP latency win (no batch contention), same stack as the conc-32/conc-8 runs. Single-stream decode tok/s + TPOT is where spec-decode helps most.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-1]
status: done
prefill_toks: 3.35
decode_toks: 16.85
mem_gb: 107.59
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: ≈60% avg draft acceptance (small N≈20 sample, noisy 53–69%) · mean acceptance length ~2.9 · per-position ~0.88/0.72/0.47 (num_speculative_tokens=3)
measured_on: 2026-06-23
completed_at: 2026-06-23 10:48 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-1 single-stream latency point (200 prompts / 300 s cap).
  scripts/bench-vllm-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 1 200 300 256 \
    --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  # only 20/200 prompts in the 300 s window (single stream + 515 s cold load). TTFT 336.6 ms, TPOT 51.5 ms.
---

**conc-1 (single-stream) point of the Qwen3.6-27B NVFP4 + MTP sweep.** Best-case MTP latency — no batch
contention.

- **Result (conc 1):** decode **16.85** tok/s single-stream; TTFT median 336.6 ms, **TPOT median 51.5 ms**
  (≈19 tok/s/stream — bandwidth-bound for a 27B at ~4-bit on GB10). Peak mem 107.6 GB.
- **Small-sample caveat:** only **20 prompts** completed in the 300 s window (single stream + a 515 s cold
  load left little time), so acceptance is **noisy** — windows ranged 53–69%, mean accept-len ~2.6–3.1.
  Treat as directional; the conc-8/conc-32 runs (134/986 prompts) are the reliable acceptance numbers
  (~67–71%). The single-stream TPOT above is the solid metric here.
