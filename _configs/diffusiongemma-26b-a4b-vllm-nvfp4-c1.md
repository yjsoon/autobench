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
status: done
prefill_toks: 132.05
decode_toks: 122.68
mem_gb: 110.23
mem_source: system MemAvailable delta (10s sampling) — NVFP4 MoE + diffusion bidirectional-attention KV
measured_on: 2026-06-23
completed_at: 2026-06-23 21:39 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
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

- **Result (conc 1):** decode **122.68 tok/s**, prefill 132.05 tok/s; **484/1000, 0 errors**, hit the
  900 s cap. Peak mem **110.23 GB**. TTFT median 1.85 s; TPOT median 0.0 (diffusion artifact — decode
  is in 256-token parallel blocks, not per-token, so client TPOT is meaningless; trust aggregate tok/s).
- **The block-diffusion mechanism is clearly visible in the engine metrics:** single-stream the server
  reported `Committed token throughput ~153 tok/s`, **~16.3 tokens committed per denoising step**, ~15.7
  denoising steps per canvas. So each forward "denoising step" commits ~16 tokens in parallel — that's
  the diffusion speedup mechanism, and it's why a single user gets ~123 tok/s here vs only **~6
  tok/s/stream** under the conc-32 queue.
- **Single-stream vs the >1100 tok/s claim:** NVIDIA's headline is a peak micro-benchmark; on real
  ShareGPT chat single-stream we measure **~123 tok/s aggregate / ~153 committed** — an order of
  magnitude below the claim. The claim does not reflect a real serving workload on this box.
- **conc-1 (123) vs conc-32 (183 aggregate):** batching still raises *aggregate* throughput, but conc-1
  gives a single user **~20× the per-stream rate** (123 vs ~6 tok/s) and a 1.85 s TTFT vs 36 s at
  conc-32. So this model's value is **low-concurrency latency** — confirmed: it's a single-user/latency
  engine, not a throughput-server engine. See [conc-8](diffusiongemma-26b-a4b-vllm-nvfp4-c8) for the
  crossover.
