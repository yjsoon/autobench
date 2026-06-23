---
title: Qwen3.6-35B-A3B · SGLang · NVFP4
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: SGLang
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) on SGLang — preferred over unsloth per policy (use the nvidia image when one exists). Base (non-speculative); MTP in the -mtp sibling. Cross-engine compare vs the vLLM NVFP4 run. NOTE — nvidia documents only a vLLM path, so SGLang support for the ModelOpt NVFP4 format must be verified at run time.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed
run_command: |
  # UNBLOCKED via the newer SGLang nightly (transformers 5.8.1 loads the qwen3.6 arch — the spark image
  # can't). Still to verify: whether SGLang's compressed-tensors path accepts nvidia's ModelOpt NVFP4
  # packing (the 27B unsloth NVFP4 loaded fine; nvidia modelopt may differ). If SGLang rejects it, BLOCK.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
    scripts/bench-sglang-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 --trust-remote-code
---

**Queued (UNBLOCKED via SGLang nightly) — Qwen3.6-35B-A3B NVFP4 base on SGLang.** Cross-engine partner to
the vLLM NVFP4 MoE base run. The stock `spark` image couldn't load the qwen3.6 arch; the
**`nightly-dev-cu13-20260623`** image (transformers 5.8.1) can — confirmed on the 27B sibling
(`qwen3-6-27b-nvfp4-sglang`, now done).

- **Repo — NVIDIA official:** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
  (ModelOpt v0.44.0), per policy (prefer nvidia where it exists).
- **Remaining risk to verify at run time:** the 27B run used **unsloth** compressed-tensors NVFP4, which
  SGLang loaded; this is **nvidia ModelOpt** NVFP4 (different packing). If SGLang's compressed-tensors path
  rejects the ModelOpt format, record the error and BLOCK (the unsloth 35B-A3B NVFP4 would be the
  SGLang-viable fallback if so).
- **Spec-decode:** in `qwen3-6-35b-a3b-nvfp4-sglang-mtp`. This is the non-spec baseline.
