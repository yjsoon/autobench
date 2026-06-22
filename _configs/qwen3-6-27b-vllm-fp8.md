---
title: Qwen3.6-27B · vLLM · FP8
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3.6-27B-FP8). Qwen3.6-27B is the current flagship dense Qwen — per Qwen, it beats the 397B Qwen3.5 model on coding benchmarks. The novel model the model list's "qwen3-6-27b" stub actually meant (earlier mis-recovered to Qwen3-32B; that valid run is now qwen3-32b-vllm-fp8).
source_repo: Qwen/Qwen3.6-27B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
context: 65536
modalities: [text, image]
mm_served: false
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned: vllm/vllm-openai (cu130-nightly, or the tf-bumped image if qwen3_5 arch needs it).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32
---

**Queued.** The current flagship **dense** Qwen (27B) — per Qwen's results it **beats the Qwen3.5-397B**
on coding benchmarks, which is why it replaces that 3.5 giant in the queue. Runtime risk to verify: the
`qwen3_5` / `Qwen3_5ForConditionalGeneration` arch is very new and multimodal — may need the
transformers-bumped vLLM image (as `gemma4_unified` did); will be benchmarked text-only. Paired with a
native-MTP config (`qwen3-6-27b-vllm-fp8-mtp`).
