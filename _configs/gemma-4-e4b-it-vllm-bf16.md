---
title: Gemma 4 E4B · vLLM · BF16
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
quant: BF16
quant_rationale: Google's BF16 base on vLLM — the full-precision reference point for the E4B, to size the FP8 speedup on a small model.
source_repo: google/gemma-4-E4B-it
download_url: https://huggingface.co/google/gemma-4-E4B-it
context: 65536
modalities: [text, image]
mm_served: false
tags: [gemma-4-e4b, Google, Gemma, BF16, ≤4B]
status: done
prefill_toks: 678.77
decode_toks: 565.78
mem_gb: 107.55
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 12:23 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). BF16 base.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly google/gemma-4-E4B-it \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model google/gemma-4-E4B-it \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The BF16 reference for the E4B — fast and clean, but FP8 still adds half again on top.** Google's
elastic Gemma-4-E4B base (BF16) on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **391 s** — clean full run, no
  time cap. Loaded in 392 s.
- **Throughput (aggregate, conc 32):** prefill **678.8 tok/s**, decode **565.8 tok/s**. TTFT median
  **158 ms**, TPOT median **52.8 ms** (≈19 tok/s/stream), req throughput 2.56/s.
- **Even a 4B model is bandwidth-bound enough for FP8 to matter:**

  | Quant | decode | prefill |
  |---|---|---|
  | BF16 | 565.8 | 678.8 |
  | FP8 | 869.7 | 1047.2 |

  FP8 is **1.54× the BF16 decode** on the same tiny model — the quant tax persists all the way down to
  4B. (It's a smaller multiple than the heavier models' ~2×, because at this size compute starts to
  share the bottleneck with bandwidth.) Both vLLM runs crush the llama.cpp Q4_K_M (435 decode): for
  small models the engine gap dominates, and within vLLM the quant gap stacks on top.
- **Memory: 107.6 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (BF16 weights ≈ 8 GB). The reservation is the same regardless of the model's tiny real size.
- **E4B is the standard `gemma4` arch** (unlike the 12B's gemma4_unified), so it serves on the stock
  vLLM with no transformers bump. Text path (`mm_served: false`).
