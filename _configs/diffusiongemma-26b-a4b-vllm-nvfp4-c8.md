---
title: DiffusionGemma-26B-A4B · vLLM · NVFP4 · conc 8
model: nvidia/diffusiongemma-26B-A4B-it
company: NVIDIA
family: Gemma
params: 25.2B / 3.8B (MoE, diffusion)
engine: vLLM
quant: NVFP4
quant_rationale: Same NVIDIA NVFP4 (ModelOpt) discrete-diffusion Gemma-4 MoE, at concurrency 8 — the mid-point of the conc 1/8/32 sweep, to locate where block-diffusion's single-stream latency advantage gives way to queue-bound throughput.
source_repo: nvidia/diffusiongemma-26B-A4B-it-NVFP4
download_url: https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 8
tags: [diffusiongemma-26b-a4b, NVIDIA, Gemma, NVFP4, 16-40B, conc-8]
status: pending
run_command: |
  scripts/bench-vllm-serving.sh nvidia/diffusiongemma-26B-A4B-it-NVFP4 65536 8 1000 900 256 \
    --trust-remote-code --attention-backend TRITON_ATTN \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 8 (wrapper defaults).
---

**Mid-point of the DiffusionGemma concurrency sweep.** Between [conc-1](diffusiongemma-26b-a4b-vllm-nvfp4-c1)
(single-stream, the block-diffusion best case) and [conc-32](diffusiongemma-26b-a4b-vllm-nvfp4)
(decode 183 tok/s aggregate, TTFT 36 s, queue-bound). conc-8 should show whether 8-way batching still
leaves the GB10 enough headroom for the parallel-block decode to stay latency-competitive.

<!-- results pending -->
