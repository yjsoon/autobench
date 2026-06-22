---
title: Gemma 4 12B · llama.cpp · Q4_K_M + MTP · conc 8
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: llama.cpp
speculative: MTP (Google assistant drafter)
quant: Q4_K_M
quant_rationale: unsloth Q4_K_M base + Google's official MTP drafter (merged GGUF). The only working path for the gemma-4-12B Google drafter — the 12B is gemma4_unified, unservable on stock vLLM/SGLang, but llama.cpp runs it via GGUF and supports --spec-type draft-mtp.
source_repo: unsloth/gemma-4-12b-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-12b-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 8
tags: [gemma-4-12b, Google, Gemma, Q4_K_M, 5-15B, conc-8]
status: done
prefill_toks: 191.35
decode_toks: 145.48
mem_gb: 25.26
mem_source: system MemAvailable delta (10s sampling) — base Q4_K_M + Q8_0-MTP draft, full KV at 65536 ctx
measured_on: 2026-06-23
completed_at: 2026-06-23 00:12 +08
engine_image: ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:12b288d6271e8de14412d61f641ca3ecd83bd73e1c4f4f22d86b2536f2b2f8e2
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744 (--spec-type draft-mtp). Base + MTP drafter (unsloth).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-12b-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 8 -cb \
    --model-draft /models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-12b-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Conc-8 point for the Gemma 4 12B MTP sweep — acceptance holds steady at ~3.15 across concurrency,
confirming the "acceptance is workload-driven, not concurrency-driven" rule (the inverse of the gpt-oss
EAGLE3 behavior).** unsloth Q4_K_M base + Google's official Q8_0 MTP drafter on llama.cpp, `-fa on`,
ctx 65536, conc 8.

- **Load:** ready in **29 s**.
- **Workload:** ShareGPT V3, concurrency 8. **179/500 completed, 0 errors** before the **300 s time cap**
  (`hit_time_cap=true`).
- **Throughput:** prefill **191.35 tok/s**, decode **145.48 tok/s** aggregate (~18.2 tok/s/stream). TTFT
  median **994 ms**, TPOT median **41.8 ms** — real latencies.
- **MTP acceptance — steady at ~3.15.** Run-aggregate **mean acceptance length 3.15**, **per-position
  (0.766, 0.580, 0.448, 0.354)** — essentially unchanged from conc-1 (3.21 / 0.789…). This is the textbook
  MTP behavior CLAUDE.md describes: acceptance is set by how well the draft predicts the *workload*, so it
  barely moves between conc-1 and conc-8. (Per-request accept-len swung 2.6–4.1 by prompt, but the running
  mean was rock-steady.) 0 errors — no draft-induced corruption, unlike gpt-oss EAGLE3.
- **Memory: 25.3 GB** = base Q4_K_M (~7.1 GB) + Q8_0 MTP draft (~0.5 GB) + 8-way KV at 65536 ctx — true
  footprint.
- Compare decode + TPOT against the conc-32 run to see how the speculative gain scales as the batch
  empties: per-stream decode 18.2 (c8) vs 49.2 (c1) — the spec gain per stream is largest at low batch,
  while aggregate throughput is highest at high batch, the expected trade-off.
