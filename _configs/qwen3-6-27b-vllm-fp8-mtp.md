---
title: Qwen3.6-27B · vLLM · FP8 + MTP
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
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned: base Qwen3.6-27B-FP8 + native MTP via vLLM --speculative-config (method finalized at run
  # time — qwen3.6 ships mtp.safetensors in-repo; vLLM exposes Qwen MTP, e.g. method "qwen3_next_mtp").
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**Queued — the speculative-decoding config for Qwen3.6-27B.** Spec accelerator lookup: Qwen3.6 ships a
**native MTP** head (`mtp.safetensors` in the base repo; unsloth also mirrors it as
`Qwen3.6-27B-MTP-GGUF`, ~880k dl/mo), so no separate EAGLE3 draft is needed — vLLM drives the in-model
MTP. Like the Gemma/llama.cpp MTP runs, expect the conc-32 gain to be smaller than single-stream, but
vLLM's batch-aware spec path could still lift decode. Exact `method` name confirmed at run time.
