---
title: Qwen3.5-35B-A3B · vLLM · AWQ · Strix Halo · BLOCKED (memory detection)
model: cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
company: Alibaba
family: Qwen
params: 35B / 3.3B (MoE, 256 experts top-8, hybrid attn+mamba)
engine: vLLM
speculative:
quant: AWQ
quant_rationale: same AWQ-MoE checkpoint that serves on SGLang here — attempted on vLLM for a same-model three-engine comparison. Blocked before quant/kernel selection by a memory-detection wall (below), so this is not an AWQ-MoE verdict.
source_repo: cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
download_url: https://huggingface.co/cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit
context: 8192
modalities: [text]
mm_served: false
concurrency: 32
tags: [qwen3.5-35b-a3b, Alibaba, Qwen, AWQ, 16-40B, conc-32, strix-halo]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-07-14
engine_image: kyuz0/vllm-therock-gfx1151:stable (vLLM 0.19.2rc1.dev, ROCm 7.13, gfx1151)
run_command: |
  # Fails at EngineCore init (determine_available_memory), NOT at quant/kernel selection.
  scripts/sweep-vllm.sh cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit qwen35-a3b-awq-vllm 8192 "1 8 32"
  # ValueError: Free memory on device cuda:0 (7.22/15.49 GiB) on startup is less than desired
  # GPU memory utilization (0.5, 7.74 GiB). Decrease GPU memory utilization or reduce GPU memory
  # used by other processes.
---

**vLLM runs on gfx1151 but only sees 15.5 GiB of the 96 GiB pool — so a 23 GiB model can't load.**
This is a **BIOS-UMA-vs-vLLM mismatch, not an AWQ or incompatibility problem.**

- **What happened:** vLLM 0.19.2 (kyuz0 TheRock gfx1151 image) starts, detects ROCm, loads the
  tokenizer, then dies in `determine_available_memory` with
  `Free memory on device cuda:0 (7.22/15.49 GiB) … less than desired`.
- **Root cause:** this box's BIOS allocates the 96 GiB UMA as **dedicated VRAM**, leaving only
  **15.5 GiB GTT** (`mem_info_gtt_total`). `torch.cuda.get_device_properties().total_memory`
  reports the **GTT** (15.49 GiB), and vLLM sizes its whole budget from that — so even
  `--gpu-memory-utilization 1.0` gives ≤15.5 GiB, below the model's ~23 GiB weights. Meanwhile
  `torch.cuda.mem_get_info()` **free** correctly shows **95.78 GiB** idle. vLLM gates on the wrong
  API.
- **Worse in practice:** of the 15.49 GiB GTT, only **~7.2 GiB is free** (the rest is held by the
  amdgpu driver / other GPU consumers), so even a 3B BF16 (~6 GiB) won't fit at useful settings —
  only the Qwen3-0.6B smoke test (~1.2 GiB) served.
- **Why the other engines are fine:** llama.cpp (Vulkan) allocates into the 96 GiB VRAM directly;
  SGLang sizes from `mem_get_info` free — both use the full pool
  ([SGLang ran this exact model](qwen3-5-35b-a3b-awq-sglang-strix-c32)).
- **Fixes (either):** (1) **BIOS** — reduce the dedicated-VRAM UMA split so GTT is large (the
  layout ROCm/vLLM expect on Strix Halo); llama.cpp/Vulkan don't need this, so it's a trade-off.
  (2) A vLLM patch to detect total memory via `mem_get_info` rather than `device_properties`.
- **Verdict:** on the current 96 GiB-VRAM / 15.5 GiB-GTT BIOS config, vLLM **serves (smoke-tested)
  but is not practically usable** — it can't fit models beyond ~a few GiB. For the whole 3B–35B
  range benchmarked here, use llama.cpp Vulkan or SGLang. See
  `notes/INCOMPATIBILITIES.md` → vLLM (Strix Halo).
