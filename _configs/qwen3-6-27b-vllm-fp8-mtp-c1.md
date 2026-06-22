---
title: Qwen3.6-27B · vLLM · FP8 + MTP · conc 1
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: FP8
quant_rationale: Qwen3.6-27B FP8 + the model's own native MTP module (mtp.safetensors ships in the base repo) — built-in multi-token-prediction speculative decoding, no separate draft.
source_repo: Qwen/Qwen3.6-27B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B, conc-1]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
measured_on:
completed_at:
run_command: |
  # base Qwen3.6-27B-FP8 + native MTP (mtp.safetensors ships in-repo) via vLLM --speculative-config.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-27B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256

# SpecDecoding metrics (vLLM, steady-state): Mean acceptance length ~3.0, Avg draft acceptance ~67%,
# per-position acceptance 0.84 / 0.67 / 0.51 (num_speculative_tokens=3).
---

**Queued — concurrency-1 variant of [Qwen3.6-27B · vLLM · FP8 + MTP].** Low-concurrency point of the Qwen3.6 native-MTP sweep (cap 500 prompts / 300 s, latency characterization). Compare decode + acceptance against the conc-32 run to see how the MTP gain scales as the batch empties.
