---
title: gpt-oss-20b · SGLang · MXFP4
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: SGLang
quant: MXFP4
quant_rationale: gpt-oss's native FP4 format, served from the original safetensors by SGLang.
source_repo: openai/gpt-oss-20b
download_url: https://huggingface.co/openai/gpt-oss-20b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe]
status: done
prefill_toks: 359.68
decode_toks: 279.49
mem_gb: 111.96
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation, see Notes
measured_on: 2026-06-21
completed_at: 2026-06-21 22:08 +08
run_command: |
  # lmsysorg/sglang:spark (the documented SOTA engine for gpt-oss on DGX Spark)
  docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/tiktoken_encodings:/tiktoken_encodings \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-20b --host 0.0.0.0 --port 30000 \
    --context-length 65536 --reasoning-parser gpt-oss --tool-call-parser gpt-oss
  python3 scripts/bench-serving.py --base-url http://localhost:30000 \
    --model openai/gpt-oss-20b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The headliner run, on its recommended engine.** llama.cpp can't serve gpt-oss's harmony
chat format (see the `gpt-oss-20b · llama.cpp` config — blocked), so this is the real number.
SGLang is the documented SOTA for gpt-oss on the DGX Spark
([LMSYS](https://www.lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/)).

- **Workload:** ShareGPT V3, 1000-entry / 15-min cap. Completed **1000/1000 with 0 errors** in
  **859 s** at **concurrency 32** — SGLang's paged KV handles every prompt length, so none of the
  long-prompt drops the llama.cpp slot-split caused (0 vs 26 errors there).
- **Throughput (aggregate, conc 32):** prefill **359.7 tok/s**, decode **279.5 tok/s**. Per-stream:
  TTFT median **304 ms**, TPOT median **83 ms** (≈12 tok/s/stream under 32-way load; the LMSYS
  ~70 tok/s figure is single-stream).
- **Memory caveat — 112 GB is a reservation, not the footprint.** SGLang pre-allocates a static
  fraction of unified memory for its KV pool (`--mem-fraction-static`, default ≈0.9 → ~109 GB of
  121 GB), so the MemAvailable delta captures that reservation, not gpt-oss-20b's intrinsic ~12 GB
  MXFP4 weights. SGLang is memory-greedy by default on unified memory; tune `--mem-fraction-static`
  to free headroom. This is why the SGLang `mem_gb` is not directly comparable to llama.cpp's.
- **Setup cost:** first launch took **400 s** to ready (downloads the full `openai/gpt-oss-20b`
  repo ~26 GB to the HF cache, then loads + warms up). Subsequent launches are cache-fast.
- Requires the tiktoken encodings (`o200k_base`, `cl100k_base`) mounted at
  `TIKTOKEN_ENCODINGS_BASE`, plus `--reasoning-parser gpt-oss --tool-call-parser gpt-oss`.
