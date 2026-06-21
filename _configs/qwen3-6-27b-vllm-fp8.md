---
title: Qwen3-32B (dense) · vLLM · FP8
model: Qwen/Qwen3-32B
company: Alibaba
family: Qwen
params: 32.8B (dense)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3-32B-FP8) — near-BF16 quality at half the bytes.
source_repo: Qwen/Qwen3-32B-FP8
download_url: https://huggingface.co/Qwen/Qwen3-32B-FP8
context: 40960
modalities: [text]
mm_served: true
tags: [Alibaba, Qwen, FP8, 16-40B]

status: done
prefill_toks: 166.44
decode_toks: 156.12
mem_gb: 105.00
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 07:40 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). ctx=40960 = model's native max.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3-32B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model Qwen/Qwen3-32B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The clean dense-vs-MoE control: same lab, same quant, same engine — half the decode of its MoE
sibling.** Alibaba's dense Qwen3-32B (32.8B), official FP8 on vLLM.

- **Repo note:** the stub's `Qwen3.6-27B` is a not-yet-real name; the current real dense Qwen3 in that
  slot is **Qwen3-32B** (32.8B), benchmarked here.
- **Workload:** ShareGPT V3, concurrency 32. **584/1000, 0 errors** — clean, but **hit the 15-min cap**.
  Loaded + CUDA-graph captured in **345 s**.
- **Throughput (aggregate, conc 32):** prefill **166.4 tok/s**, decode **156.1 tok/s**. TTFT median
  **649 ms**, TPOT median **179 ms** (≈5.6 tok/s/stream), req throughput 0.62/s.
- **Dense pays the full-parameter tax.** Against **Qwen3-30B-A3B** (the MoE sibling — same FP8, vLLM,
  ctx 40960): MoE decode **331** vs this dense **156**, a clean **~2.1×** gap. The MoE activates only
  ~3B params per token; the dense model fires all 32.8B, so per-token compute is ~10× the active
  weight and decode roughly halves. This is the sharpest dense-vs-sparse comparison in the set because
  every other variable is held fixed. It also lands just under the dense Qwen2.5-Coder-32B llama.cpp
  run for the same reason (size-bound, not quant-bound).
- **Memory: 105.0 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (FP8 weights ≈ 33 GB).
- **Context:** native max 40960; benchmarked at that.
