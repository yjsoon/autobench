---
title: Gemma 4 E4B · vLLM · FP8
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
quant: FP8
quant_rationale: leon-se's FP8-Dynamic (compressed-tensors) — no trusted NVFP4 exists for the E-series, and FP8 is the well-used (329k dl/mo) trusted quant. The E4B is the standard gemma4 arch, so it serves on the stock vLLM.
source_repo: leon-se/gemma-4-E4B-it-FP8-Dynamic
download_url: https://huggingface.co/leon-se/gemma-4-E4B-it-FP8-Dynamic
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, FP8, ≤4B, conc-32]
status: done
prefill_toks: 1047.23
decode_toks: 869.66
mem_gb: 109.86
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 11:00 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). leon-se FP8-Dynamic.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly leon-se/gemma-4-E4B-it-FP8-Dynamic \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model leon-se/gemma-4-E4B-it-FP8-Dynamic \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The fastest decode in the entire benchmark — the tiny E4B on vLLM's FP8 kernels.** Google's elastic
Gemma-4-E4B (~4B effective), leon-se FP8-Dynamic on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in just **253 s** — the quickest
  full run of any config. Loaded in 240 s.
- **Throughput (aggregate, conc 32):** prefill **1047.2 tok/s**, decode **869.7 tok/s** — **the highest
  decode measured anywhere in this benchmark** (next best is the Granite-3B Mamba hybrid at 617). TTFT
  median **102 ms**, TPOT median **33.8 ms** (≈30 tok/s/stream), req throughput 3.94/s.
- **vLLM FP8 doubles the llama.cpp path for this small model:**

  | Engine / quant | prefill | decode |
  |---|---|---|
  | llama.cpp Q4_K_M | 329.0 | 435.0 |
  | **vLLM FP8** | **1047.2** | **869.7** |

  At conc 32 the GB10 has plenty of compute headroom for a ~4B model, and vLLM's batched FP8 MoE/attention
  kernels exploit it far better than llama.cpp's server — **2× the decode and 3.2× the prefill** on the
  identical model. For small models specifically, the engine gap is enormous; the heavy giants (where
  memory bandwidth dominates) show much smaller engine gaps.
- **The E-series stays light and is the standard `gemma4` arch** (not the 12B's `gemma4_unified`), so it
  serves on the stock vLLM with no transformers bump. Quant note: no trusted NVFP4 exists for the
  E-series, so FP8 (leon-se, 329k downloads/mo) is the chosen 4-bit-class point.
- **Memory: 109.9 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (FP8 weights ≈ 4 GB — the reservation dwarfs the actual model for a model this small).
- Text path benchmarked (`mm_served: false`).
