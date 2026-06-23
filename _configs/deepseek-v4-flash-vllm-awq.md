---
title: DeepSeek V4-Flash · vLLM · AWQ-Int4
model: deepseek-ai/DeepSeek-V4-Flash
company: DeepSeek
family: DeepSeek
params: 284B (MoE) — measured via llama.cpp (was mislabeled 158B)
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

**Blocked — corrected 2026-06-23: the real wall is FIT, and the engine now exists.**

Two facts changed since the 2026-06-22 note:
1. **`deepseek_v4` IS now in vLLM** — `nightly-aarch64` (vLLM 0.23.1, the cu130-nightly successor) ships
   `DeepseekV4ForCausalLM` + `deepseek_eagle3` + a GB10-capable NVFP4-MoE oracle. So "arch not in the
   engines" is **no longer the blocker**.
2. **The NVFP4 repo does NOT fit.** `nvidia/DeepSeek-V4-Flash-NVFP4` is **168.3 GB** (its own
   `safetensors.index.json`; `MIXED_PRECISION` = FP8 backbone + NVFP4 experts, ≈ FP8-sized — the "~79 GB"
   was a nominal-4-bit guess). 168 GB **> the 121 GB ceiling** → won't load on one Spark at all.

A *fitting* build (~80 GB) means a **true** 4-bit quant: GGUF Q4 (needs a WIP/community `deepseek_v4`
llama.cpp fork — not upstream) or an INT4-AWQ/GPTQ community repo on vLLM (trust caveat). Despite this
page's title, **no NVFP4 build fits**; the AWQ-int4 route is the realistic one if a trusted repo is
chosen. See `deepseek-v4-flash-vllm-nvfp4-eagle3` for the full corrected investigation. Stays blocked
pending a fitting 4-bit build.

**Re-verified 2026-06-23 (user re-requested `nvidia/DeepSeek-V4-Flash-NVFP4`):** `model.safetensors.index.json`
`total_size` = **168 266 793 544 bytes ≈ 168.3 GB** — confirmed > 121 GB. NVIDIA's own model card serves
it **multi-GPU** (`vllm ... --tensor-parallel-size 4` on GB300; SGLang `--tensor-parallel-size 8`),
corroborating that it is not a single-GB10 target. The card lists **no MTP/EAGLE** for this model
(the `deepseek_v4` engine *does* ship `deepseek_eagle3`, tracked in `deepseek-v4-flash-vllm-nvfp4-eagle3`,
also blocked on the same FIT wall). **So an MTP variant can't be run either — the base doesn't load on one
Spark.** Would need a true ~80 GB 4-bit build to revisit.
