---
title: Phi-4-reasoning-vision 15B · vLLM · FP8
model: microsoft/Phi-4-reasoning-vision-15B
company: Microsoft
family: Phi
params: 15B (dense)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: microsoft/Phi-4-reasoning-vision-15B
download_url: https://huggingface.co/microsoft/Phi-4-reasoning-vision-15B
context: 131072
modalities: [text, image]
mm_served: true
tags: [phi-4-reasoning-vision-15b, Microsoft, Phi, FP8, 5-15B]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # blocked — needs a vision benchmark, not this text-only harness (see Notes)
---

**Blocked — this model needs a different benchmark than the rest of the suite.** Decision 2026-06-22.

`Phi-4-reasoning-vision-15B` is a **vision-language reasoning model** (`modalities: [text, image]`) whose
core purpose is reasoning over images. Every other config here is measured with the **text-only**
methodology — ShareGPT chat prompts driven through `/v1/chat/completions`, reporting prefill/decode
tok/s at concurrency 32. That harness sends **no images**, so:

- **A text-only run would not exercise the vision tower at all** — it would produce a prefill/decode
  number that ignores the entire image-encoding + cross-attention path that defines this model. For a
  vision-*first* model that's misleading, not just incomplete (unlike the Gemma multimodal models,
  where text is a primary use case and a text-path number is fair).
- **A representative benchmark requires a different setup:** an image dataset (e.g. a VQA / document set),
  multimodal request payloads (`image_url` content parts), and image-aware throughput metrics
  (image-prefill time, tokens-per-image, decode under vision context). Those numbers would **not be
  comparable** to any text-only config in this benchmark — it's effectively a separate methodology.

**To unblock:** stand up a small vision-serving benchmark (multimodal `bench-serving.py` variant +
image workload) and run it as its own track, or intentionally record a text-only number with a loud
caveat. Left blocked until that scope decision is made. The model serves fine on vLLM FP8
(`microsoft/Phi-4-reasoning-vision-15B`); the blocker is methodology, not availability.
