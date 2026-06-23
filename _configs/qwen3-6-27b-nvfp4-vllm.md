---
title: Qwen3.6-27B · vLLM · NVFP4
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4), calibrated on HF UltraChat @16K. NVFP4 is a genuine GB10 fast-path — beat the official FP8 base by ~21% decode here. Base (non-speculative) config; the repo's MTP module is exercised in the -mtp sibling.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 181.8
decode_toks: 187.74
mem_gb: 108.18
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85)
measured_on: 2026-06-23
completed_at: 2026-06-23 09:54 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # NVFP4 (modelopt_fp4 auto-detected from the unsloth checkpoint) on vLLM nightly-aarch64. Base, conc-32.
  scripts/bench-vllm-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --dtype bfloat16
  # 683/1000 prompts, 0 errors, hit the 900 s time cap (the cold load took 648 s — download + compile —
  # leaving < 5 min of the window for prompts). TTFT median 796.9 ms, TPOT median 155.1 ms.
---

**The NVFP4 fast-path pays off on the 27B too — ~21% faster decode than the official FP8 base.** Unsloth
NVFP4 (W4A4) of Qwen3.6-27B, served text-only on vLLM (the card's recommended path), at conc-32.

- **Result (conc 32):** decode **187.7** tok/s vs the [FP8 base]'s **154.7** (**+21%**); prefill 181.8 vs
  168.9. Peak mem **108.2 GB** (vLLM static KV reservation at util 0.85), ~same as FP8 (107.5) — NVFP4
  saves on weights but the KV reservation dominates the headline. **0 errors.**
- **Time cap:** the run **hit the 900 s cap at 683/1000 prompts** — the cold start took **648 s**
  (first-time ~15 GB download + torch.compile), leaving little of the window. The tok/s is steady-state
  and comparable, but the entry count is capped; the MTP/SGLang siblings reuse the cached weights and run
  the full count. (Flagged per the run-cap policy.)
- **Repo choice — unsloth (no NVIDIA option).** Policy prefers an official `nvidia/` NVFP4 when one
  exists, but **NVIDIA publishes none for the 27B** (only for the 35B-A3B sibling); the 27B has only
  community NVFP4 quants (unsloth, mmangkad, sakamakismile, …). So this uses **unsloth** (a trusted,
  well-known quantizer per the repo policy).
- **Pair:** base (this) + `qwen3-6-27b-nvfp4-vllm-mtp` (with MTP). SGLang siblings under `*-sglang*`.
