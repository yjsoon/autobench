---
title: gpt-oss-120b · vLLM · MXFP4 + EAGLE3
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: vLLM
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss MXFP4 base + NVIDIA's throughput-tuned EAGLE3 draft head (nvidia/gpt-oss-120b-Eagle3-throughput) — the spec-decode dimension on the gpt-oss-120b headliner. gpt-oss is text-only, so it should dodge vLLM's multimodal draft-model block that complicated Gemma.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned: vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, NVIDIA EAGLE3-throughput draft.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"model":"nvidia/gpt-oss-120b-Eagle3-throughput","method":"eagle3","num_speculative_tokens":3}'
---

**Queued.** EAGLE3 spec-decode on the gpt-oss-120b headliner, to compare against the base
gpt-oss-120b vLLM run (279/253) and the SGLang run (188/140). Uses NVIDIA's **throughput**-optimized
EAGLE3 head — the variant matched to 32-way serving (vs the `-short-context`/`-long-context`/latency
heads). Expectation: a vLLM batch-aware EAGLE3 lift like Gemma saw (+41–59% at conc 32), though the
harmony-parse errors from 256-token truncation will persist (the draft only speeds decode, it doesn't
change the chat-path parsing). Will run after the current 41-130B MoE-giant sweep.
