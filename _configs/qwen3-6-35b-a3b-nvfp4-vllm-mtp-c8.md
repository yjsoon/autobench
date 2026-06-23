---
title: Qwen3.6-35B-A3B · vLLM · NVFP4 + MTP · conc 8
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: Concurrency-8 point of the Qwen3.6-35B-A3B NVFP4 + native-MTP sweep — same stack as the conc-32 run (nvidia ModelOpt NVFP4 base on marlin + in-repo MTP head on triton), lower batch for the latency-characterization point. Acceptance should hold ~constant vs conc-32 (workload-driven); what changes is the per-stream TPOT.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 8
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-8]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-8 latency point (500 prompts / 300 s cap, matching the -c8 convention). Same NVIDIA Spark
  # MTP recipe as the conc-32 run.
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 8 500 300 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3 --moe-backend marlin \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
---

**conc-8 point of the Qwen3.6-35B-A3B NVFP4 + MTP sweep.** Lower-batch latency characterization
(cap 500 prompts / 300 s). Pairs with the conc-32 done run (decode 541 tok/s) and the conc-1 sibling.

- **What to capture:** aggregate prefill/decode tok/s, per-stream TPOT, peak mem, and the SpecDecoding
  acceptance — expect mean accept-len ~3.0 to hold (workload-driven, as on the 27B sweep).
- **TPOT caveat:** the `qwen3` reasoning-parser zeroes client TPOT median — read the in-engine
  SpecDecoding metrics + aggregate decode tok/s instead.
