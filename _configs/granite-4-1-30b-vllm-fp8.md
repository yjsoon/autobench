---
title: Granite 4.1 30B · vLLM · FP8
model: ibm-granite/granite-4.1-30b
company: IBM
family: Granite
params: 28.9B (MoE)
engine: vLLM
quant: FP8
quant_rationale: IBM's official FP8 (compressed-tensors) weights — near-BF16 quality at half the bytes, Apache-2.0.
source_repo: ibm-granite/granite-4.1-30b-FP8
download_url: https://huggingface.co/ibm-granite/granite-4.1-30b-FP8
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [granite-4.1-30b, IBM, Granite, FP8, 16-40B, conc-32]
status: done
prefill_toks: 220.82
decode_toks: 181.81
mem_gb: 102.36
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 07:00 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly ibm-granite/granite-4.1-30b-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model ibm-granite/granite-4.1-30b-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**IBM's 30B Granite MoE on vLLM — clean, but a more compute-heavy MoE than the 3B-active speedsters.**
Granite 4.1 30B (28.9B MoE), official FP8, Apache-2.0.

- **Workload:** ShareGPT V3, concurrency 32. **765/1000, 0 errors** — clean (no parse failures), but
  **hit the 15-min cap** before the 1000th prompt. Loaded + CUDA-graph captured in **280 s**.
- **Throughput (aggregate, conc 32):** prefill **220.8 tok/s**, decode **181.8 tok/s**. TTFT median
  **481 ms**, TPOT median **158 ms** (≈6.3 tok/s/stream), req throughput 0.82/s.
- **Not all MoEs are 3B-active.** Decode (182) lands well below the Qwen3-Coder-30B-A3B (296) and the
  NVFP4 Nemotron 30B-A3B models (353–389) on the same vLLM/GB10 setup. Those activate only ~3B/token;
  Granite-4.1-30B's MoE uses a **larger active fraction**, so per-token compute is higher and decode is
  closer to a dense mid-size model than to the sparse-active speedsters. It hit the time cap for the
  same reason. The clean 0-error run shows the model itself serves fine on vLLM — this is a
  compute-profile difference, not a problem.
- **Memory: 102.4 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (FP8 weights ≈ 29 GB). The hybrid-Mamba memory advantage seen in the small **GGUF** Granites
  (3B at 16.9 GB, 8B at 25.4 GB on llama.cpp) can't be read off a vLLM run, since vLLM pre-reserves the
  KV pool regardless.
- **Context:** native 131072; benchmarked at 65536 for cross-config comparability.
