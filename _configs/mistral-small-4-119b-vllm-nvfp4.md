---
title: Mistral Small 4 119B · vLLM · NVFP4
model: mistralai/Mistral-Small-4-119B-2603-NVFP4
company: Mistral AI
family: Mistral
params: 119B (dense)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4, hardware-accelerated on GB10. Mistral's OWN official NVFP4 build of Small-4-119B (the base repo is FP8; this is the separate -NVFP4 checkpoint) — trusted source.
source_repo: mistralai/Mistral-Small-4-119B-2603-NVFP4
download_url: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4
context: 131072
modalities: [text, image]
mm_served: true
tags: [Mistral AI, Mistral, NVFP4, 41-130B]

status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned configuration — filled in by the run when benchmarked
---

Stub — not yet benchmarked. **Repo verified 2026-06-22:** `mistralai/Mistral-Small-4-119B-2603-NVFP4`
exists, ungated, 13 safetensors (Mistral's own org → trusted). It's a `Mistral3ForConditionalGeneration`
multimodal model (vision tower); we benchmark text-only ShareGPT. Dense 119B → expect slow decode,
in the gpt-oss-120b / Nemotron-Super-120B class. Run on `vllm/vllm-openai:cu130-nightly`.
