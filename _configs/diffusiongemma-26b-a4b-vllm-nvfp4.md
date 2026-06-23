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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). Discrete-diffusion Gemma MoE, NVFP4 (ModelOpt), vLLM + --trust-remote-code
  # (loads the custom diffusion arch from the repo). The card's example pins --max-num-seqs 4; we target
  # the conc-32 sweep but may need to drop to 4/8 if the diffusion scheduler OOMs or rejects high batch.
  scripts/bench-vllm-serving.sh nvidia/diffusiongemma-26B-A4B-it-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --attention-backend TRITON_ATTN \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --enable-auto-tool-choice
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 32 (wrapper defaults)
---

**Queued — DiffusionGemma-26B-A4B NVFP4 on vLLM (novel diffusion-LM datapoint).** A discrete-diffusion
transformer (encoder-decoder, bidirectional attention) that generates in **parallel 256-token blocks**
rather than autoregressively — NVIDIA claims **>1100 tok/s**. NVFP4 (ModelOpt) of NVIDIA's diffusion
variant of Google's Gemma-4-26B-A4B.

- **Repo:** [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4),
  Apache-2.0 (+ Gemma terms), 25.2B total / 3.8B active MoE. Served via vLLM, `--trust-remote-code`
  (custom diffusion arch), forced `TRITON_ATTN` backend, `gemma4` reasoning/tool parsers.
- **No spec-decode:** the diffusion block-parallel decode *is* the speedup mechanism — there's no MTP/
  EAGLE config, so this is a single base page (no `-mtp` sibling).
- **Throughput-measurement caveat:** decode happens in 256-token parallel blocks, so the usual
  autoregressive **prefill vs decode tok/s split may not map cleanly** — record aggregate output tok/s
  and note the block-diffusion behaviour (and whether bench-serving's streaming counts blocks sanely)
  in the result Notes. Native context 262K; benchmarked at 65536.
- **Concurrency:** the card example uses `--max-num-seqs 4`; if conc-32 OOMs or the diffusion scheduler
  balks, drop to 4/8 and record the working value.
- **At run time:** confirm `nightly-aarch64` loads the custom diffusion arch (it may require a specific
  vLLM/transformers version shipped with the repo's remote code); if it rejects the arch, record it and
  BLOCK rather than guess.
