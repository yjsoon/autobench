---
title: Qwen3.6-35B-A3B · vLLM · FP8 + MTP
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
speculative: MTP
quant: FP8
quant_rationale: Qwen3.6-35B-A3B FP8 + the model's native MTP module (mtp.safetensors ships in-repo) — built-in multi-token-prediction speculative decoding on the sparse MoE.
source_repo: Qwen/Qwen3.6-35B-A3B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
context: 65536
modalities: [text, image]
mm_served: false
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned: base Qwen3.6-35B-A3B-FP8 + native MTP via vLLM --speculative-config (method finalized at
  # run time; qwen3.6 ships mtp.safetensors in-repo).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
---

**Queued — the speculative-decoding config for the Qwen3.6 MoE.** Native **MTP** head ships in the base
(`mtp.safetensors`; unsloth `Qwen3.6-35B-A3B-MTP-GGUF`, ~900k dl/mo). On a 3B-active MoE the base decode
is already fast, so — per the Gemma/12B MTP findings — the conc-32 spec-decode gain may be modest; the
in-model MTP + vLLM's batch-aware path is the best shot at a lift. Method name confirmed at run time.
