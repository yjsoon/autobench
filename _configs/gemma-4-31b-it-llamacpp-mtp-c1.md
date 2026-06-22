---
title: Gemma 4 31B · llama.cpp · Q4_K_M + MTP · conc 1
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: llama.cpp
speculative: MTP (Google assistant drafter)
quant: Q4_K_M
quant_rationale: unsloth Q4_K_M base + Google's official MTP drafter (merged GGUF) — the only working path to benchmark Google's gemma-4 assistant drafter (SGLang's spark image has no gemma4 support; vLLM rejects gemma multimodal draft-model spec-decode).
source_repo: unsloth/gemma-4-31B-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [gemma-4-31b, Google, Gemma, Q4_K_M, 16-40B, conc-1]
status: done
prefill_toks: 28.31
decode_toks: 23.96
mem_gb: 36.15
mem_source: system MemAvailable delta (10s sampling) — base Q4_K_M + Q8_0-MTP draft, full KV at 65536 ctx
measured_on: 2026-06-23
completed_at: 2026-06-23 00:18 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744 (has --spec-type draft-mtp; MTP merged 2026-06-07).
  # Base + Google MTP drafter both under /home/gauravmm/models (unsloth merged GGUFs).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-31B-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 1 -cb \
    --model-draft /models/MTP/gemma-4-31B-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-31B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Conc-1 point for the Gemma 4 31B MTP sweep — the highest MTP acceptance in the whole Gemma set
(mean accept-len ~3.41).** unsloth Q4_K_M base + Google's official Q8_0 MTP drafter on llama.cpp,
`-fa on`, ctx 65536, conc 1.

- **Load:** ready in **38 s**.
- **Workload:** ShareGPT V3, concurrency 1. **29/500 completed, 0 errors** before the **300 s time cap** —
  the 31B is the slowest Gemma, so only ~29 single-stream requests finish in 5 min.
- **Throughput:** decode **23.96 tok/s** (single stream), prefill **28.31 tok/s**. TTFT median **528 ms**,
  TPOT median **34.2 ms**.
- **MTP acceptance — best in the Gemma set.** Run-aggregate **mean acceptance length 3.41**, **per-position
  (0.813, 0.644, 0.517, 0.413)** — the largest dense base predicts its own continuations best, so the
  first draft slot lands ~81% and even the 4th ~41%. Per-request accept-len reached ~4.0 on predictable
  prompts. 0 errors. Acceptance climbs monotonically with model size across the MTP runs (E4B 2.76 →
  12B 3.21 → 31B 3.41), the expected "bigger model ⇒ better self-draft" trend.
- **Memory: 36.2 GB** = base Q4_K_M (~18.3 GB) + Q8_0 MTP draft (~0.5 GB) + full KV at 65536 ctx — true
  footprint (no static KV pre-reservation on llama.cpp).
- Compare decode + TPOT against the conc-32 run: at conc-1 the highest accept-len means the per-stream
  speedup is largest here, while the conc-32 run trades that for aggregate throughput.
