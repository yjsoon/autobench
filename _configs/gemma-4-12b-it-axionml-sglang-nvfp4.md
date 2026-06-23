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
status: pending
run_command: |
  # AxionML documents SGLang for this checkpoint, but requires a Gemma-4-capable image/branch
  # with ModelOpt FP4 support (the card calls out transformers >= 5.10 handling for Gemma-4).
  # Verify the exact image/tag before running.
  SGLANG_IMAGE=<gemma4-modelopt-capable-sglang-image> \
    scripts/bench-sglang-serving.sh AxionML/Gemma-4-12B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt_fp4 --kv-cache-dtype fp8_e4m3 \
    --reasoning-parser gemma4 --tool-call-parser gemma4 --mem-fraction-static 0.85
---

**Queued — AxionML's dense Gemma-4-12B NVFP4 on the engine its card actually documents.**

- **Engine choice:** queueing **SGLang**, not vLLM first. AxionML's card gives explicit SGLang launch
  flags for this quant and a separate speculative recipe on the same stack; that is a stronger source
  than the generic auto-generated vLLM snippet on the Hugging Face page.
- **Quant details:** the card says this is an **MLP-only NVFP4 recipe** following NVIDIA's dense
  Gemma-4 approach: FFN in NVFP4, attention kept in BF16, KV cache in FP8. Claimed size is
  **~11 GB vs ~24 GB BF16**, which should be a good fit on Spark.
- **Text-only benchmarking:** the base model is any-to-any, but the harness workload is ShareGPT text,
  so this config is marked `mm_served: false`.
- **Spec sibling:** pair with `gemma-4-12b-it-axionml-sglang-nvfp4-mtp` using the official
  `google/gemma-4-12B-it-assistant` drafter via SGLang's **NEXTN** path.

