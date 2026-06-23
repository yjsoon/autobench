---
title: Gemma 4 12B · SGLang · NVFP4 · AxionML
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: SGLang
quant: NVFP4
quant_rationale: AxionML/Gemma-4-12B-NVFP4 — a user-requested community NVFP4 dense-Gemma checkpoint with a detailed model card and explicit SGLang deployment guidance. Queue it on SGLang because the card documents that path directly, including the ModelOpt FP4 and KV-cache settings needed for Gemma-4 on Blackwell.
source_repo: AxionML/Gemma-4-12B-NVFP4
download_url: https://huggingface.co/AxionML/Gemma-4-12B-NVFP4
context: 65536
modalities: [text, image, audio, video]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, NVFP4, 5-15B, conc-32]
status: done
prefill_toks: 446.48
decode_toks: 386.56
mem_gb: 108.43
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation (mem-fraction-static 0.85)
measured_on: 2026-06-23
completed_at: 2026-06-23 23:54 +0800
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # AxionML's June 2026 recipe is not a stock image tag: it requires a newer SGLang nightly,
  # a branch checkout (`bzhng-development/sglang@gemma4-modelopt-ptq`), and transformers 5.10-dev.
  # The model-card example tag `nightly-dev-20260604-14ed9b44` no longer existed on Docker Hub, so
  # this run used the newer cu13 nightly plus the same documented overlay.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
  SGLANG_BRANCH=gemma4-modelopt-ptq \
  TRANSFORMERS_REF=1423d22f7a3b62e8c70ad67b58ec25cd9b675897 \
    scripts/bench-sglang-gemma4-modelopt-serving.sh AxionML/Gemma-4-12B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --mem-fraction-static 0.85
  # 1000/1000 prompts, 0 errors, 603.6 s. TTFT median 210.9 ms, TPOT median 75.7 ms.
---

**DONE — Axion's documented SGLang path does run on Spark, but only with the full branch overlay and it
lands well behind the trusted Red Hat AI vLLM NVFP4 baseline.**

- **Result (conc 32):** prefill **446.48** tok/s, decode **386.56** tok/s aggregate; **1000/1000, 0
  errors** in **603.6 s**. Median **TTFT 210.9 ms**, median **TPOT 75.7 ms**, request throughput
  **1.657 req/s**.
- **Engine path actually required:** the queued placeholder was real. Axion's model card does not map to
  a stock published SGLang image; it requires a newer nightly, a checkout of
  **`bzhng-development/sglang@gemma4-modelopt-ptq`**, and **transformers 5.10-dev**. The card's example
  image tag **`nightly-dev-20260604-14ed9b44`** was already gone from Docker Hub, so this run used the
  newer **`nightly-dev-cu13-20260623-ba9d5aed`** base and applied the same documented overlay inside the
  container.
- **Load behavior:** server readiness took **296 s**. The branch recognized the model as
  **`Gemma4UnifiedForConditionalGeneration`**, switched to **`ModelOptModelLoader`**, and loaded the
  already-quantized checkpoint directly with **`quantization=modelopt_fp4`** and **FP8 KV cache**.
- **Comparison vs the other 12B NVFP4 path:** this is materially slower than the already-done
  **RedHatAI/gemma-4-12B-it-NVFP4 on vLLM nightly** (**386.56 vs 503.81 decode tok/s**, about **23%**
  lower). So Axion's dense-Gemma NVFP4 recipe is runnable, but it is not the throughput leader on this
  ShareGPT conc-32 benchmark.
- **Memory:** **108.43 GB** headline usage, again dominated by the engine's static reservation rather
  than the checkpoint alone. The "11 GB model" claim from the card is compatible with that: most of the
  observed footprint here is still context-serving overhead at 65k/32-way concurrency.
