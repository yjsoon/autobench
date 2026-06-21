---
title: Qwen3-30B-A3B · vLLM · FP8
model: Qwen/Qwen3-30B-A3B
company: Alibaba
family: Qwen
params: 30.5B / 3B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3-30B-A3B-FP8) — near-BF16 quality at half the bytes, FP8-accelerated MoE on Blackwell.
source_repo: Qwen/Qwen3-30B-A3B-FP8
download_url: https://huggingface.co/Qwen/Qwen3-30B-A3B-FP8
context: 40960
modalities: [text]
mm_served: true
tags: [Alibaba, Qwen, FP8, 16-40B]

status: done
prefill_toks: 336.23
decode_toks: 331.28
mem_gb: 101.36
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 07:18 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). ctx=40960 = model's native max.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3-30B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model Qwen/Qwen3-30B-A3B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The fastest FP8 MoE in the benchmark — clean 1000/1000, second only to the NVFP4 Nemotrons.**
Alibaba's general-purpose Qwen3-30B-A3B (30.5B total / 3B active), official FP8 on vLLM.

- **Repo note:** the stub's `Qwen3.6-35B-A3B` is a not-yet-real name; the current real model in that
  slot is **Qwen3-30B-A3B** (30.5B / 3B MoE), benchmarked here — a model-list-name recovery.
- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **761 s** — clean full run, no
  time cap. Loaded + CUDA-graph captured in **301 s**.
- **Throughput (aggregate, conc 32):** prefill **336.2 tok/s**, decode **331.3 tok/s**. TTFT median
  **330 ms**, TPOT median **90 ms** (≈11 tok/s/stream), req throughput 1.31/s.
- **Edges out its own Coder sibling.** Qwen3-Coder-30B-A3B (also 3B-active FP8) decoded 296; this
  general model hits **331**. Part of the gap is context: this run used the model's native **40960** max
  vs the Coder's 65536, so the per-token KV is smaller and the decode batch a touch cheaper. Both are
  clean 0-error runs and both sit in the **FP8-MoE tier below the NVFP4 Nemotron 30B-A3B models
  (353–389)** — consistent with NVFP4 > FP8 for raw decode on Blackwell at the same ~3B active.
- **Memory: 101.4 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (FP8 weights ≈ 30 GB).
