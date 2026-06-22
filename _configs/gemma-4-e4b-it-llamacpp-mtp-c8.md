---
title: Gemma 4 E4B · llama.cpp · Q4_K_M + MTP · conc 8
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
concurrency: 8
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B, conc-8]
status: done
prefill_toks: 300.29
decode_toks: 222.52
mem_gb: 14.90
mem_source: system MemAvailable delta (10s sampling) — base Q4_K_M + Q8_0-MTP draft, full KV at 65536 ctx
measured_on: 2026-06-22
completed_at: 2026-06-22 23:59 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744. NOTE: -fa off (flash-attn) is REQUIRED here —
  # the E-series + MTP draft crashes the GB10 flash-attn kernel (ggml-cuda/fattn.cu:110 fatal error)
  # with -fa on OR default/auto; only -fa off loads. (The 12B/31B MTP runs use -fa on fine.)
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 8 -cb \
    --model-draft /models/MTP/gemma-4-E4B-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa off \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-E4B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Conc-8 point for the Gemma 4 E4B MTP sweep — the MTP draft works *cleanly* here (unlike the gpt-oss
EAGLE3 heads), with healthy ~2.76 mean acceptance length and zero errors.** unsloth Q4_K_M base +
Google's official Q8_0 MTP drafter on llama.cpp, `-fa off` (E-series + MTP crashes the GB10 flash-attn
kernel), ctx 65536, conc 8.

- **Load:** ready in **22 s**.
- **Workload:** ShareGPT V3, concurrency 8. **271/500 completed, 0 errors** before the **300 s time cap**
  (`hit_time_cap=true`).
- **Throughput:** prefill **300.29 tok/s**, decode **222.52 tok/s** aggregate (~27.8 tok/s/stream). TTFT
  median **585 ms**, TPOT median **28.6 ms** — real latencies (no harmony buffering on llama.cpp).
- **MTP acceptance — healthy and as-expected (the contrast with gpt-oss EAGLE3).** Run-aggregate
  **mean acceptance length 2.76** (with `--spec-draft-n-max 4`), **per-position acceptance
  (0.672, 0.472, 0.350, 0.271)** — i.e. the first drafted token lands ~67% of the time and acceptance
  decays smoothly down the four draft slots, exactly the MTP shape CLAUDE.md expects (~70% first-pos,
  mean ~2.x–3.0). Per-request draft acceptance ranged ~0.34–0.53. This is the **opposite** of the
  gpt-oss EAGLE3 collapse (~1.0–1.7 mean, harmony corruption): the Google MTP head genuinely predicts
  the Gemma chat stream. Acceptance held steady across the run (not concurrency-degrading).
- **Memory: 14.9 GB** = base Q4_K_M (~4.8 GB) + Q8_0 MTP draft (~0.1 GB) + full KV at 65536 ctx — true
  footprint (llama.cpp doesn't pre-reserve a static KV pool like vLLM), so this is directly comparable to
  the non-spec E4B llama.cpp runs.
- Compare decode + TPOT against the conc-32 run to see how the speculative gain scales as the batch
  empties.
