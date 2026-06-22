---
title: gpt-oss-120b · vLLM · MXFP4 + EAGLE3 · conc 8
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
concurrency: 8
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-8]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, NVIDIA EAGLE3-throughput draft.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-120b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"model":"nvidia/gpt-oss-120b-Eagle3-throughput","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Queued — concurrency-8 variant of [gpt-oss-120b · vLLM · MXFP4 + EAGLE3].** Low-concurrency point for the spec-decode concurrency sweep (the run is otherwise identical; cap reduced to 500 prompts / 300 s since low-conc runs are latency characterizations, not throughput-to-1000). Compare decode + TPOT against this config's conc-32 run to see how the speculative gain scales as the batch empties.
