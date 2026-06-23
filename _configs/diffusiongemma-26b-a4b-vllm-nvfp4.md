---
title: DiffusionGemma-26B-A4B · vLLM · NVFP4
model: nvidia/diffusiongemma-26B-A4B-it
company: NVIDIA
family: Gemma
params: 25.2B / 3.8B (MoE, diffusion)
engine: vLLM
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/diffusiongemma-26B-A4B-it-NVFP4, ModelOpt) of a discrete-diffusion Gemma-4 MoE — a genuinely different decode mechanism (parallel 256-token blocks, claimed >1100 tok/s) worth a throughput datapoint on GB10. NVIDIA-built diffusion variant of Google's Gemma-4-26B-A4B base; vLLM is the documented serving path.
source_repo: nvidia/diffusiongemma-26B-A4B-it-NVFP4
download_url: https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [diffusiongemma-26b-a4b, NVIDIA, Gemma, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 211.39
decode_toks: 183.2
mem_gb: 118.15
mem_source: system MemAvailable delta (10s sampling) — NVFP4 MoE + diffusion bidirectional-attention KV (near the 121 GB ceiling)
completed_at: 2026-06-23 13:31 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # Discrete-diffusion Gemma MoE, NVFP4 (ModelOpt), vLLM + --trust-remote-code (loads the custom
  # DiffusionGemmaForBlockDiffusion arch from the repo). conc-32 ran cleanly — no OOM, no batch rejection.
  scripts/bench-vllm-serving.sh nvidia/diffusiongemma-26B-A4B-it-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --attention-backend TRITON_ATTN \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 32 (wrapper defaults).
  # 742/1000 prompts (hit 900 s cap), 0 errors. ready after 432 s. quantization=modelopt_fp4,
  # NvFp4 MoE backend FLASHINFER_CUTLASS. TTFT median 35943 ms (very high — see notes).
---

**DONE — DiffusionGemma-26B-A4B NVFP4 on vLLM (novel diffusion-LM datapoint).** A discrete-diffusion
transformer that generates in **parallel 256-token blocks** rather than autoregressively — NVIDIA claims
**>1100 tok/s**. Loads and serves cleanly on `nightly-aarch64`; the conc-32 ShareGPT serving number is
**far below** that headline (which is a single-stream best-case, not a 32-way server queue).

- **Result (conc 32):** prefill 211.4 / decode **183.2** tok/s aggregate; **742/1000** prompts (hit the
  900 s cap), **0 errors**; peak mem **118.15 GB** — near the 121 GB ceiling. **TTFT median 35.9 s** (!).
- **Loads cleanly:** `nightly-aarch64` resolved the custom arch `DiffusionGemmaForBlockDiffusion`,
  `quantization=modelopt_fp4`, NvFp4 MoE backend `FLASHINFER_CUTLASS`, forced `TRITON_ATTN`. No
  arch-rejection, no OOM at conc-32, no batch-size rejection — so the card's `--max-num-seqs 4` is
  conservative; 32 ran fine.
- **>1100 tok/s claim is single-stream best-case, NOT a serving number.** Under conc-32 ShareGPT the
  block-diffusion scheduler keeps **only ~5–18 of 32 requests running** at a time (rest queued) at very
  low **KV-cache usage (0.6–1.0%)**, with generation throughput oscillating ~150–285 tok/s and prefill
  bursting to 2000+ tok/s. Net **183 tok/s decode aggregate** — slower than the autoregressive
  Gemma-4/Qwen3.6 MoE NVFP4 runs at the same concurrency. The diffusion parallelism helps *per-request
  latency on a free GPU*, not *aggregate server throughput on a saturated queue*.
- **Very high TTFT (36 s median):** with the queue 20–29 deep and each request decoding in 256-token
  blocks, time-to-first-token is dominated by queue wait — this model is built for low-concurrency,
  latency-sensitive single-stream generation, not batched serving.
- **Metric caveat:** decode happens in 256-token parallel blocks, so the autoregressive prefill/decode
  split and client TPOT (reads 0.0, also zeroed by the `gemma4` reasoning-parser) don't map cleanly —
  trust the **aggregate output tok/s** (183) and the in-engine generation-throughput log. Native context
  262K; benchmarked at 65536.
- **No spec-decode:** the diffusion block-parallel decode *is* the speedup mechanism — no MTP/EAGLE
  sibling.
- **Repo:** [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4),
  Apache-2.0 (+ Gemma terms), 25.2B total / 3.8B active MoE.
