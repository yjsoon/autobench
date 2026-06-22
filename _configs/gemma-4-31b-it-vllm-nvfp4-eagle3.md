---
title: Gemma 4 31B · vLLM · NVFP4 + EAGLE3
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: vLLM
speculative: EAGLE3
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + RedHatAI's official EAGLE3 speculator — fast quant plus lossless speculative decoding, the combination that finally clears the 31B's time cap.
source_repo: nvidia/Gemma-4-31B-IT-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B]
status: done
prefill_toks: 307.7
decode_toks: 264.72
mem_gb: 107.60
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 10:51 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly. NVFP4 base + RedHatAI EAGLE3 speculator (draft model).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-31B-IT-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-31B-IT-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**EAGLE3 finally clears the 31B's time cap — the biggest speculative win in the set: +59% decode, and
the run completes all 1000 prompts.** Google's dense Gemma-4-31B, NVIDIA NVFP4 base + RedHatAI's
EAGLE3 speculator, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **876 s** — the **first full
  completion** of the dense 31B at any quant (Q4_K_M, FP8, and NVFP4 all hit the cap). Loaded +
  CUDA-graph captured in **394 s**.
- **Throughput (aggregate, conc 32):** prefill **307.7 tok/s**, decode **264.7 tok/s**. TTFT median
  **706 ms**, TPOT median **108.8 ms** (≈9.2 tok/s/stream), req throughput 1.14/s.
- **The 31B full ladder — engine × quant × speculation:**

  | Config | decode | completed | time-cap |
  |---|---|---|---|
  | Q4_K_M / llama.cpp | 78.5 | 322/1000 | yes |
  | FP8 / vLLM | 147.6 | 603/1000 | yes |
  | NVFP4 / vLLM | 167.0 | 672/1000 | yes |
  | **NVFP4 + EAGLE3 / vLLM** | **264.7** | **1000/1000** | **no** |

  EAGLE3 adds **~1.59× decode over the NVFP4 base** (167 → 265) — a *larger* speculative gain than the
  26B MoE saw (1.41×). Dense models benefit more from EAGLE3 here: each verified step on a
  compute-bound dense model amortizes a full expensive forward over multiple accepted draft tokens,
  whereas the already-cheap MoE step has less to amortize. The combined NVFP4+EAGLE3 stack takes the
  31B from **78 tok/s and 322/1000 (the worst run in the benchmark)** to **265 tok/s and a clean
  1000/1000** — a **3.4× decode improvement** end to end, all lossless.
- **Spec-decode path:** vLLM EAGLE3 with the RedHatAI speculator (the only spec-decode route that works
  for Gemma 4 on the stock images — SGLang's spark image has no `gemma4` support; see the 26B EAGLE3
  page).
- **Memory: 107.6 GB is the vLLM `--gpu-memory-utilization 0.85` reservation** (NVFP4 weights ≈ 17 GB +
  the EAGLE3 head), not the footprint.
- Text path benchmarked (`mm_served: false`).
