---
title: Qwen3.6-35B-A3B · vLLM · FP8
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3.6-35B-A3B-FP8). The sparse-MoE 3.6 model — 3B active, agentic-coding focused; per Qwen it dramatically beats its 3.5 predecessor and is the natural replacement for the Qwen3.5-122B-A10B MoE.
source_repo: Qwen/Qwen3.6-35B-A3B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # planned: vllm/vllm-openai (cu130-nightly, or tf-bumped image if qwen3_5_moe arch needs it).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32
---

**Queued.** The **sparse-MoE** Qwen3.6 (35B total / 3B active) — fast (3–4× the dense 27B per Qwen) and
the agentic-coding workhorse. Replaces the blocked Qwen3.5-122B-A10B MoE in the queue. Same `qwen3_5_moe`
multimodal-arch runtime risk as the 27B (may need the tf-bumped image; benchmarked text-only). Paired
with a native-MTP config (`qwen3-6-35b-a3b-vllm-fp8-mtp`).
