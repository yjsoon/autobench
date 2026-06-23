---
title: DiffusionGemma-26B-A4B · vLLM · NVFP4 · conc 1
model: nvidia/diffusiongemma-26B-A4B-it
company: NVIDIA
family: Gemma
params: 25.2B / 3.8B (MoE, diffusion)
engine: vLLM
quant: NVFP4
quant_rationale: Same NVIDIA NVFP4 (ModelOpt) discrete-diffusion Gemma-4 MoE as the conc-32 run, at concurrency 1 — the regime this model is built for. The conc-32 notes found block-diffusion parallelism helps per-request latency on a free GPU, not aggregate throughput on a saturated queue; conc-1 tests that claim (NVIDIA cites >1100 tok/s single-stream).
source_repo: nvidia/diffusiongemma-26B-A4B-it-NVFP4
download_url: https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [diffusiongemma-26b-a4b, NVIDIA, Gemma, NVFP4, 16-40B, conc-1]
status: pending
run_command: |
  # Same path as conc-32 (model already cached); concurrency 1 — the single-stream regime where the
  # 256-token block-diffusion decode should pay off (vs conc-32's 36 s TTFT / 183 tok/s queued).
  scripts/bench-vllm-serving.sh nvidia/diffusiongemma-26B-A4B-it-NVFP4 65536 1 1000 900 256 \
    --trust-remote-code --attention-backend TRITON_ATTN \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 1 (wrapper defaults).
---

**Single-stream point for DiffusionGemma — the regime block-diffusion is built for.** Companion to the
[conc-32 run](diffusiongemma-26b-a4b-vllm-nvfp4) (decode 183 tok/s aggregate, TTFT 36 s, only ~5–18 of
32 requests ever running). That run concluded the diffusion parallelism helps *per-request latency on a
free GPU*, not aggregate throughput on a saturated queue — so conc-1, with the whole GB10 to itself and
no queue wait, is where the 256-token parallel-block decode should approach NVIDIA's >1100 tok/s claim.

<!-- results pending — runs after the gpt-oss-120b SGLang+EAGLE3 conc-32 run frees the GPU -->
