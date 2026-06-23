---
title: Qwen3.6-27B · vLLM · NVFP4 + MTP
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4) + the checkpoint's own MTP module — "this checkpoint includes the MTP module, so it can act as its own speculative draft" (model card). Native multi-token-prediction spec-decode, no separate draft.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 283.3
decode_toks: 274.07
mem_gb: 108.53
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 67% avg draft acceptance · mean acceptance length 3.0 · per-position 0.84/0.66/0.51 (num_speculative_tokens=3)
measured_on: 2026-06-23
completed_at: 2026-06-23 10:19 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # NVFP4 base + native MTP (Resolved architecture: Qwen3_5MTP) on vLLM nightly-aarch64. conc-32.
  scripts/bench-vllm-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  # 986/1000 prompts, 0 errors (hit 900 s cap by a hair). TTFT median 958 ms, TPOT median 104.7 ms.
  # SpecDecoding (steady-state): mean acceptance length ~3.0, avg draft acceptance ~67%,
  # per-position 0.84 / 0.66 / 0.51 (num_speculative_tokens=3).
---

**The fastest 27B config in the sweep — NVFP4 + native MTP, decode 274 tok/s at conc-32.** Unsloth NVFP4
base + the in-repo MTP module on vLLM.

- **Result (conc 32):** prefill **283.3** / decode **274.07** tok/s aggregate; **986/1000, 0 errors**
  (just grazed the 900 s cap). TTFT median 958 ms, TPOT median 104.7 ms. Peak mem **108.5 GB**.
- **MTP speedup:** vs the [NVFP4 base] (decode 187.7) that's **+46%**; vs the stock [FP8 + MTP] conc-32
  (decode ~241) it's **+14%** — NVFP4 stacks cleanly on top of MTP. This is the fastest 27B decode here.
- **Acceptance: ~67% avg draft acceptance, mean accept-len ~3.0** (per-position 0.84 / 0.66 / 0.51,
  `num_speculative_tokens=3`). **Cross-check ✓:** this *matches* the stock FP8+MTP run's ~67% / ~3.1 on
  ShareGPT — so the NVFP4 quant does **not** degrade the MTP head's acceptance (no red flag). Right at the
  published MTP expectation for general chat (the ~70–85% band is coding-skewed; ShareGPT runs a touch
  lower).
- **Pair:** base `qwen3-6-27b-nvfp4-vllm` + this (MTP). conc-8 / conc-1 variants in
  `qwen3-6-27b-nvfp4-vllm-mtp-c8` / `-c1`. SGLang siblings under `*-sglang*`.
- **Repo choice — unsloth (no NVIDIA option).** Policy prefers an official `nvidia/` NVFP4 when one
  exists, but NVIDIA publishes none for the 27B (only the 35B-A3B sibling); 27B has only community NVFP4
  quants. Using **unsloth** (trusted quantizer per policy).
