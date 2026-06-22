---
title: Gemma 4 E4B · llama.cpp · Q4_K_M + MTP · conc 1
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: llama.cpp
speculative: MTP (Google assistant drafter)
quant: Q4_K_M
quant_rationale: unsloth Q4_K_M base + Google's official MTP drafter (merged GGUF) — the Google assistant drafter on llama.cpp, the only engine that runs it (vLLM rejects gemma spec-decode; sglang:spark has no gemma4).
source_repo: unsloth/gemma-4-E4B-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-E4B-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 1
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B, conc-1]
status: done
prefill_toks: 121.9
decode_toks: 99.6
mem_gb: 11.12
mem_source: system MemAvailable delta (10s sampling) — base Q4_K_M + Q8_0 MTP draft + 65536-ctx KV
measured_on: 2026-06-22
completed_at: 2026-06-22 21:10 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744. NOTE: -fa off (flash-attn) is REQUIRED here —
  # the E-series + MTP draft crashes the GB10 flash-attn kernel (ggml-cuda/fattn.cu:110 fatal error)
  # with -fa on OR default/auto; only -fa off loads. (The 12B/31B MTP runs use -fa on fine.)
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 1 -cb \
    --model-draft /models/MTP/gemma-4-E4B-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa off \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-E4B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Concurrency-1 (latency) point of the Gemma 4 E4B · llama.cpp · Q4_K_M + MTP spec-decode sweep.** ShareGPT, 500-prompt / 300 s cap.

- **Throughput:** prefill **121.9 tok/s**, decode **99.6 tok/s** (~65 tok/s/stream class beaten — single stream here). TTFT median **231 ms**, TPOT median **7.7 ms**, request throughput 0.40 req/s.
- **Run hit the 300 s time cap** at **119 / 500** prompts completed, 0 errors — flagged per policy (slow path relative to the 1000-entry target; expected for a single-stream latency characterization).
- **Peak memory 11.12 GB** (system MemAvailable delta) — base Q4_K_M + Q8_0 MTP draft + 65536-ctx KV, well within the 121 GB ceiling.

**MTP draft acceptance (the reason to run this config):**
- Aggregate over the run: **mean accepted length ≈ 2.88** tokens/step (with `--spec-draft-n-max 4`), per-position acceptance **(0.718, 0.504, 0.374, 0.285)** — i.e. ~72% of first drafted tokens accepted, tapering by position. Overall accepted/generated draft-token rate ≈ **0.47**.
- This sits **below the ~67% ShareGPT figure seen on Qwen3.6-27B MTP** but is not a red flag: the drafter here is Google's small E4B *assistant* MatFormer drafter on a heavily-quantized (Q4_K_M) base over general chat, so a first-position ~0.72 / mean-len 2.88 is plausible and consistent across tasks (per-task draft acceptance swung 0.41–0.53, no concurrency-driven instability since conc=1). Compare against the conc-8 run to confirm acceptance stays flat with batch and to see where the spec speedup materializes.

> `-fa off` is **required** for the E-series + MTP draft — flash-attn crashes the GB10 kernel (`ggml-cuda/fattn.cu:110`) with `-fa on`/auto; only `-fa off` loads. (The 12B/31B MTP runs use `-fa on` fine.)
