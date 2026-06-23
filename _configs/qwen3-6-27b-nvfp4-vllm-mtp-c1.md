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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # conc-1 single-stream latency point (200 prompts / 300 s cap).
  scripts/bench-vllm-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 1 200 300 256 \
    --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**conc-1 (single-stream) point of the Qwen3.6-27B NVFP4 + MTP sweep.** Best-case MTP latency — no batch
contention, so the accepted draft tokens translate most directly into a per-token speedup. Compare the
single-stream decode tok/s against the NVFP4 base at conc-1 to read the pure MTP win; acceptance should
match the conc-32/conc-8 ~67%.
