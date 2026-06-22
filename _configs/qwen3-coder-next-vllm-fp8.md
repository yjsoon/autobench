---
title: Qwen3-Coder-Next · vLLM · FP8
model: Qwen/Qwen3-Coder-Next
company: Alibaba
family: Qwen
params: 79.7B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Near-BF16 quality at half the bytes; official FP8 weights published.
source_repo: Qwen/Qwen3-Coder-Next
download_url: https://huggingface.co/Qwen/Qwen3-Coder-Next
context: 262144
modalities: [text]
mm_served: true
tags: [qwen3-coder-next, Alibaba, Qwen, FP8, 41-130B]
status: done
prefill_toks: 203.79
decode_toks: 184.71
mem_gb: 105.40
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 16:54 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3-Coder-Next-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model Qwen/Qwen3-Coder-Next-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**An 80B coder MoE — clean, mid-pack decode, the largest Qwen MoE benchmarked.** Alibaba's
Qwen3-Coder-Next (79.7B total MoE), official FP8 on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **753/1000, 0 errors** — clean (no parse failures), but
  **hit the 15-min cap**. Loaded + CUDA-graph captured in **428 s**.
- **Throughput (aggregate, conc 32):** prefill **203.8 tok/s**, decode **184.7 tok/s**. TTFT median
  **618 ms**, TPOT median **167 ms** (≈6 tok/s/stream), req throughput 0.81/s.
- **Where it lands among the MoEs:** decode **185** is well below the small 3B-active MoEs
  (Qwen3-30B-A3B 331, Qwen3-Coder-30B-A3B 296) — this "Next" model activates more parameters per token
  than the A3B line, so per-token compute is higher. It sits near the dense Gemma-31B NVFP4 (167) and
  the granite-30B MoE (182), consistent with a larger active fraction. Clean 0-error run shows the
  model serves fine on the stock vLLM.
- **Memory: 105.4 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint (the
  FP8 weights are ~40 GB). As with every vLLM/SGLang config, treat as a reservation.
- **Context:** native max 262144; benchmarked at 65536 for cross-config comparability.
