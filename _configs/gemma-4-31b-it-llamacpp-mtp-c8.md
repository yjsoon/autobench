---
title: Gemma 4 31B · llama.cpp · Q4_K_M + MTP · conc 8
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
concurrency: 8
tags: [gemma-4-31b, Google, Gemma, Q4_K_M, 16-40B, conc-8]
status: done
prefill_toks: 107.15
decode_toks: 62.22
mem_gb: 49.50
mem_source: system MemAvailable delta (10s sampling) — base Q4_K_M + Q8_0-MTP draft, full KV at 65536 ctx
measured_on: 2026-06-23
completed_at: 2026-06-23 00:24 +08
engine_image: ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:12b288d6271e8de14412d61f641ca3ecd83bd73e1c4f4f22d86b2536f2b2f8e2
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744 (has --spec-type draft-mtp; MTP merged 2026-06-07).
  # Base + Google MTP drafter both under /home/gauravmm/models (unsloth merged GGUFs).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-31B-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 8 -cb \
    --model-draft /models/MTP/gemma-4-31B-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-31B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Conc-8 point for the Gemma 4 31B MTP sweep — acceptance holds at ~3.38, steady from conc-1 (3.41).**
unsloth Q4_K_M base + Google's official Q8_0 MTP drafter on llama.cpp, `-fa on`, ctx 65536, conc 8.

- **Load:** ready in **29 s**.
- **Workload:** ShareGPT V3, concurrency 8. **79/500 completed, 0 errors** before the **300 s time cap**.
- **Throughput:** prefill **107.15 tok/s**, decode **62.22 tok/s** aggregate (~7.8 tok/s/stream). TTFT median
  **3.1 s** (8-way queue on the slow 31B), TPOT median **97.9 ms**.
- **MTP acceptance — steady at ~3.38.** Run-aggregate **mean acceptance length 3.38**, **per-position
  (0.811, 0.648, 0.510, 0.409)** — essentially unchanged from conc-1 (3.41 / 0.813…), again confirming
  acceptance is workload- not concurrency-driven. Per-request accept-len ranged 2.7–4.0. 0 errors.
- **Memory: 49.5 GB** = base Q4_K_M (~18.3 GB) + Q8_0 MTP draft (~0.5 GB) + 8-way KV at 65536 ctx — true
  footprint.
- **Closes the Gemma llama.cpp MTP sweep** (E4B / 12B / 31B × conc-1/8): every run clean (0 errors), with
  healthy MTP acceptance scaling by model size (2.76 → 3.21 → 3.41 at conc-1) and holding flat across
  concurrency — the textbook MTP profile, and the sharp contrast to the gpt-oss EAGLE3 collapse.
