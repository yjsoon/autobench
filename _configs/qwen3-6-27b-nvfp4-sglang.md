---
title: Qwen3.6-27B · SGLang · NVFP4
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: SGLang
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4) on SGLang — the card lists SGLang as a supported path. Base (non-speculative). Required a NEWER SGLang nightly (nightly-dev-cu13-20260623, transformers 5.8.1) — the stock spark image can't load the qwen3.6 arch. Cross-engine compare vs the vLLM NVFP4 run.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 179.07
decode_toks: 177.99
mem_gb: 103.21
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation; weights 24.55 GB resident (from log)
measured_on: 2026-06-23
completed_at: 2026-06-23 11:35 +08
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # Stock spark image CAN'T load qwen3.6 (transformers 4.57.1). Used a newer SGLang nightly
  # (nightly-dev-cu13-20260623-ba9d5aed, arm64, transformers 5.8.1) via the new SGLANG_IMAGE override.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
    scripts/bench-sglang-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 32 1000 900 256 --trust-remote-code
  # 647/1000 prompts, 0 errors (hit 900 s cap). ready after 276 s. TTFT median 2169 ms, TPOT median 159 ms.
  # Load log: type=Qwen3_5ForConditionalGeneration, quant=compressed-tensors, weights 24.55 GB.
---

**UNBLOCKED on a newer SGLang nightly — cross-engine NVFP4 datapoint vs vLLM.** The stock
`lmsysorg/sglang:spark` image (transformers 4.57.1) couldn't load the Qwen3.6 (`qwen3_5`) arch; the
**`nightly-dev-cu13-20260623-ba9d5aed`** image (arm64, **transformers 5.8.1**) loads it fine.

- **Result (conc 32):** prefill 179.1 / decode **177.99** tok/s aggregate; **647/1000, 0 errors** (hit the
  900 s cap). Peak mem **103.2 GB**; weights **24.55 GB** resident (log) — confirms NVFP4 is active (a BF16
  fallback would be ~54 GB and far slower).
- **Cross-engine vs vLLM:** decode **178.0 (SGLang) vs 187.7 (vLLM)** on the same unsloth NVFP4 / 65536 /
  conc-32 — **vLLM ~5% faster** on decode, SGLang ~5 GB leaner on peak mem. Close; neither dominates for
  this 27B NVFP4 base.
- **NVFP4 caveat (benign):** SGLang logs *"Acceleration for non-quantized schemes is not supported by
  Compressed Tensors. Falling back to UnquantizedLinearMethod"* — this applies only to the model's
  **non-quantized layers** (norms/embeddings); `quant=compressed-tensors` + the 24.55 GB footprint + the
  178 tok/s decode all confirm the 4-bit expert/linear weights are used.
- **Tokenizer warning:** *"Tokenizer … is still TokenizersBackend … attributes may be missing"* — did not
  cause errors (0 failed requests), noted in case it matters for a later run.
- **Engine image:** pinned the nightly via the new `SGLANG_IMAGE=` wrapper override. The blocked SGLang
  Qwen3.6 siblings (27B MTP, 35B-A3B base/MTP) are unblocked the same way.
