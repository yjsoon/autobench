---
title: Qwen3.6-35B-A3B · vLLM · NVFP4 + MTP · conc 1
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: Single-stream point of the Qwen3.6-35B-A3B NVFP4 + native-MTP sweep — same stack as the conc-32 run (nvidia ModelOpt NVFP4 base on marlin + in-repo MTP head on triton). Best-case MTP latency with no batch contention; the per-stream TPOT here is the headline number.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-1]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-1 single-stream latency point (200 prompts / 300 s cap). Same NVIDIA Spark MTP recipe.
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 1 200 300 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3 --moe-backend marlin \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
---

**conc-1 (single-stream) point of the Qwen3.6-35B-A3B NVFP4 + MTP sweep.** Best-case MTP latency — no
batch contention. Pairs with the conc-32 done run (decode 541 tok/s agg) and the conc-8 sibling.

- **What to capture:** single-stream decode tok/s + TPOT (the meaningful low-conc metric), peak mem, and
  SpecDecoding accept-len. Single stream + cold load means only a handful of prompts land in the 300 s
  window — that's expected; the per-stream latency is the point, not throughput.
- **TPOT caveat:** the `qwen3` reasoning-parser can zero the client TPOT median — corroborate with the
  in-engine SpecDecoding metrics.
