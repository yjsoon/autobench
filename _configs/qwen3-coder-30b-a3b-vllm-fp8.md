---
title: Qwen3-Coder-30B-A3B · vLLM · FP8
model: Qwen/Qwen3-Coder-30B-A3B-Instruct
company: Alibaba
family: Qwen
params: 30.5B / 3B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8) — near-BF16 quality at half the bytes, FP8-accelerated MoE on Blackwell.
source_repo: Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
download_url: https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8
context: 65536
modalities: [text]
mm_served: true
tags: [Alibaba, Qwen, FP8, 16-40B]

status: done
prefill_toks: 344.41
decode_toks: 295.82
mem_gb: 106.27
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 06:39 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The clean counter-example to the slow-MoE result: a 3B-active MoE on the right engine flies.**
Alibaba's Qwen3-Coder-30B-A3B (30.5B total / 3B active), official FP8 on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **743 s** — a perfectly clean
  full run, no time cap. Loaded + CUDA-graph captured in **288 s**.
- **Throughput (aggregate, conc 32):** prefill **344.4 tok/s**, decode **295.8 tok/s**. TTFT median
  **342 ms**, TPOT median **103 ms** (≈9.7 tok/s/stream), req throughput 1.35/s.
- **MoE speed is an engine property, not just an active-param count.** Compare directly with
  **DeepSeek-Coder-V2-Lite** (16B / 2.4B-active MoE) on **llama.cpp**, which managed only **130 decode**
  and hit the time cap: here a *larger* 30B / 3B-active MoE on **vLLM's fused FP8 MoE kernels** more than
  doubles that to **296 decode with zero errors**. Same "sparse-active MoE" class, opposite outcome —
  the difference is the engine's MoE + attention kernels (Qwen3MoE is first-class in vLLM; DeepSeek-V2
  MLA+fine-grained-MoE is not well-served by llama.cpp on GB10). It still trails the **NVFP4** Nemotron
  MoEs (353–389), consistent with FP8 < NVFP4 for raw decode on Blackwell.
- **Memory: 106.3 GB is the vLLM reservation (`--gpu-memory-utilization 0.85`), not the footprint.** The
  FP8 weights are ~30 GB; the rest is the static KV pool vLLM pre-allocates. Treat as a reservation, as
  with every vLLM/SGLang config.
- **Context:** native max is 262144; benchmarked at 65536 for cross-config comparability.
