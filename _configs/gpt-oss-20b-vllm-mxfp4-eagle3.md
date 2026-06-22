---
title: gpt-oss-20b · vLLM · MXFP4 + EAGLE3
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: vLLM
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss MXFP4 base + RedHatAI's EAGLE3 speculator (speculators format, 32k dl/mo) — the spec-decode dimension on gpt-oss-20b, pairing with its SGLang/vLLM/llama.cpp configs.
source_repo: openai/gpt-oss-20b
download_url: https://huggingface.co/openai/gpt-oss-20b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe]
status: done
prefill_toks: 883.7
decode_toks: 686.48
mem_gb: 106.51
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-22
completed_at: 2026-06-22 17:52 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, RedHatAI EAGLE3 speculator.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-20b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**EAGLE3 is a clear win on gpt-oss-20b — faster *and* cleaner — the exact mirror image of what it did
to the 120b.** gpt-oss-20b MXFP4 + RedHatAI's EAGLE3 speculator, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **350 s** — a clean full run.
  Loaded + CUDA-graph captured in **216 s**.
- **Better than the base on every axis:**

  | gpt-oss-20b config | prefill | decode | completed | errors |
  |---|---|---|---|---|
  | vLLM (base) | 654.1 | 535.3 | 892 | 108 |
  | **vLLM + EAGLE3** | **883.7** | **686.5** | **1000** | **0** |
  | SGLang (base) | 359.7 | 279.5 | 1000 | 0 |

  Decode **+28%** (535 → 686), prefill +35%, and — notably — **0 harmony errors vs the base's 108**:
  the EAGLE3 draft path changed how generation streams enough that the 256-token-truncation parse
  failures disappeared, so it also fixed the base run's robustness problem here. Fastest gpt-oss
  result recorded.
- **The size × concurrency rule, demonstrated on one model family.** Identical technique (vLLM EAGLE3,
  conc 32), opposite outcomes by size: **gpt-oss-20b +28%**, **gpt-oss-120b −45%**. The 20b (3.6B
  active) leaves the GB10 with spare compute at conc 32 for the draft to exploit; the 120b is already
  saturated, so the same draft is pure overhead. Spec-decode pays off when there's headroom —
  small model and/or low concurrency — and backfires without it.
- **Harmony caveat:** TTFT 8.8 s / TPOT 0.0 remain artifacts of the harmony chat path (buffered
  reasoning), but the aggregate tok/s are valid and clearly the best of the gpt-oss-20b configs.
