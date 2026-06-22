---
title: DeepSeek V4-Flash · vLLM · AWQ-Int4
model: deepseek-ai/DeepSeek-V4-Flash
company: DeepSeek
family: DeepSeek
params: 158B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: NVIDIA NVFP4 (~79 GB) is the only build that fits one Spark (FP8 ~160 GB). Blocked on engine support, not quant — see Notes.
source_repo: nvidia/DeepSeek-V4-Flash-NVFP4
download_url: https://huggingface.co/nvidia/DeepSeek-V4-Flash-NVFP4
context: 131072
modalities: [text]
mm_served: true
concurrency: 32
tags: [deepseek-v4-flash, DeepSeek, NVFP4, 130B+, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — deepseek_v4 arch not yet in the stock vLLM/SGLang Spark images (see Notes)
---

**Blocked — fits at NVFP4, but the `deepseek_v4` arch isn't in the stock engines yet.** Updated
2026-06-22.

The base (158B MoE) fits one Spark only at **`nvidia/DeepSeek-V4-Flash-NVFP4` (~79 GB)** — the FP8 is
~160 GB, over the ceiling (the stub's AWQ-Int4 idea was right; NVFP4 is the trusted fitting build). But
neither **vLLM cu130-nightly** nor **SGLang `:spark`** ships a `deepseek_v4` model implementation (both
stop at `deepseek_v2`/V3), so the base won't load on the documented Spark engines. See
`deepseek-v4-flash-vllm-nvfp4-eagle3` for the full investigation (done while evaluating the EAGLE3.1
speculative draft) and the unblock options (newer engine build, or llama.cpp GGUF). Unblock once
`deepseek_v4` lands in the stock images.
