---
title: Qwen3.6-35B-A3B · vLLM · FP8 · conc 1
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
concurrency: 1
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B, conc-1]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). qwen3_5_moe loads on stock image.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 1
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Queued — no-MTP baseline for Qwen3.6-35B-A3B at concurrency 1.** The matched base run so the MTP speedup at conc 1 is computable (vs `qwen3-6-35b-a3b-vllm-fp8-mtp-c1`). Cap 500 prompts / 300 s.
