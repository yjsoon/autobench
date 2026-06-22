---
title: gpt-oss-20b · vLLM · MXFP4 + EAGLE3
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: vLLM
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss MXFP4 base + RedHatAI's EAGLE3 speculator (speculators format, 32k dl/mo) — the spec-decode dimension on gpt-oss-20b, pairing with its SGLang/vLLM/llama.cpp configs.
source_repo: openai/gpt-oss-20b
download_url: https://huggingface.co/openai/gpt-oss-20b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned: vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, RedHatAI EAGLE3 speculator.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
---

**Queued.** EAGLE3 spec-decode on gpt-oss-20b, to compare against its base vLLM run (654/535), SGLang
(360/279), and the (blocked) llama.cpp harmony path. RedHatAI's speculator (same `speculators`
EAGLE3 format used for the Gemma runs). Run after the gpt-oss-120b EAGLE3 config.
