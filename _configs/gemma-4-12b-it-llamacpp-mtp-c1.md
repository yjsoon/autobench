---
title: Gemma 4 12B · llama.cpp · Q4_K_M + MTP · conc 1
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
concurrency: 1
tags: [gemma-4-12b, Google, Gemma, Q4_K_M, 5-15B, conc-1]
status: done
prefill_toks: 82.6
decode_toks: 49.24
mem_gb: 20.31
mem_source: system MemAvailable delta (10s sampling) — base Q4_K_M + Q8_0-MTP draft, full KV at 65536 ctx
measured_on: 2026-06-23
completed_at: 2026-06-23 00:06 +08
engine_image: ghcr.io/ggml-org/llama.cpp:full-cuda@sha256:12b288d6271e8de14412d61f641ca3ecd83bd73e1c4f4f22d86b2536f2b2f8e2
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744 (--spec-type draft-mtp). Base + MTP drafter (unsloth).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-12b-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 1 -cb \
    --model-draft /models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-12b-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Conc-1 point for the Gemma 4 12B MTP sweep — the strongest MTP acceptance in the Gemma set
(mean accept-len ~3.21 at single-stream).** unsloth Q4_K_M base + Google's official Q8_0 MTP drafter on
llama.cpp, `-fa on` (the 12B dense model is fine with flash-attn, unlike the E-series), ctx 65536, conc 1.

- **Load:** ready in **33 s**.
- **Workload:** ShareGPT V3, concurrency 1. **59/500 completed, 0 errors** before the **300 s time cap**
  (`hit_time_cap=true`) — at conc-1 only ~59 requests finish in 5 min.
- **Throughput:** decode **49.24 tok/s** (single stream), prefill **82.6 tok/s**. TTFT median **276 ms**,
  TPOT median **16.6 ms** — real latencies.
- **MTP acceptance — excellent and as-expected.** Run-aggregate **mean acceptance length 3.21** (with
  `--spec-draft-n-max 4`), **per-position acceptance (0.789, 0.599, 0.466, 0.356)** — the first drafted
  token lands ~79% of the time, even the 4th slot ~36%. Per-request mean accept-len climbed into the
  3.1–3.6 range. This is *higher* than the E4B (2.76) — the bigger dense base + its matched MTP head
  predict the chat stream very well, and at conc-1 (max headroom) the full draft depth pays off. Healthy
  MTP, the antithesis of the gpt-oss EAGLE3 collapse.
- **Memory: 20.3 GB** = base Q4_K_M (~7.1 GB) + Q8_0 MTP draft (~0.5 GB) + full KV at 65536 ctx — true
  footprint (no static KV pre-reservation on llama.cpp).
- Compare decode + TPOT against the conc-32 run to see how the speculative gain scales as the batch
  empties — at conc-1 the accept-len is highest, so the per-stream speedup should be largest here.
