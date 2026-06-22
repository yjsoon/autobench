---
title: Llama 4 Scout 17B-16E · vLLM · AWQ-Int4
model: meta-llama/Llama-4-Scout-17B-16E-Instruct
company: Meta
family: Llama
params: 109B / 17B (MoE)
engine: vLLM
quant: W4A16
quant_rationale: meta-llama base is gated; used RedHatAI's ungated W4A16 (compressed-tensors Int4) quant. 4-bit to fit one Spark, quality-preserving.
source_repo: RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16
download_url: https://huggingface.co/RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16
context: 65536
modalities: [text]
mm_served: false
concurrency: 32
tags: [llama-4-scout-17b-16e, Meta, Llama, W4A16, 41-130B, conc-32]
status: done
prefill_toks: 86.65
decode_toks: 61.48
mem_gb: 106.59
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 17:15 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly. meta-llama base is gated → RedHatAI ungated W4A16.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model RedHatAI/Llama-4-Scout-17B-16E-Instruct-quantized.w4a16 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Llama-4 Scout — a 109B MoE, but its 17B active params make it decode like a heavy dense model, not a
sparse speedster.** Meta's Llama-4-Scout-17B-16E (109B total / 17B active / 16 experts), RedHatAI W4A16
(the meta-llama base is gated; this ungated quant is the substitute).

- **Workload:** ShareGPT V3, concurrency 32. **271/1000, 0 errors**, **hit the 15-min cap**. Loaded +
  CUDA-graph captured in **296 s**.
- **Throughput (aggregate, conc 32):** prefill **86.7 tok/s**, decode **61.5 tok/s**. TTFT median
  **1.65 s**, TPOT median **478 ms** (≈2.1 tok/s/stream), req throughput 0.28/s.
- **High active-param count is the story.** Decode **61.5** is the slowest of any MoE here — because
  Scout activates **17B params/token** (16 experts but a large per-expert size), ~5–6× the 3B-active
  Qwen MoEs (296–331). At that active fraction it behaves like a ~17B-dense forward with MoE routing
  overhead, and lands near the dense **Llama-3.3-70B** tier (49) rather than the sparse-MoE tier. MoE
  total size doesn't predict speed — **active params do**, and Scout sits at the heavy end.
- **Memory: 106.6 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint (the
  W4A16 weights are ~55 GB). 
- **Context note:** Llama-4 advertises a very long context (up to ~10M); benchmarked at 65536 for
  cross-config comparability, not as a long-context test (that would be a separate methodology).
