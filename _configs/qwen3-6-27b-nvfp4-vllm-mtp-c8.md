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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # conc-8 latency point (500 prompts / 300 s cap, matching the FP8-MTP -c8 convention).
  scripts/bench-vllm-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 8 500 300 256 \
    --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**conc-8 point of the Qwen3.6-27B NVFP4 + MTP sweep.** Lower-batch latency characterization (500-prompt /
300 s cap). Compare per-stream TPOT + acceptance against the conc-32 run — acceptance should stay ~67%
(workload-driven), while aggregate tok/s drops with the smaller batch.
