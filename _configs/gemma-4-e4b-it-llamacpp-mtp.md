---
title: Gemma 4 E4B · llama.cpp · Q4_K_M + MTP
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
tags: [gemma-4-e4b, Google, Gemma, Q4_K_M, ≤4B]

status: done
prefill_toks: 209.81
decode_toks: 276.59
mem_gb: 17.73
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 12:58 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744. NOTE: -fa off (flash-attn) is REQUIRED here —
  # the E-series + MTP draft crashes the GB10 flash-attn kernel (ggml-cuda/fattn.cu:110 fatal error)
  # with -fa on OR default/auto; only -fa off loads. (The 12B/31B MTP runs use -fa on fine.)
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-E4B-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 32 -cb \
    --model-draft /models/MTP/gemma-4-E4B-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa off \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-E4B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The Google MTP drafter on the E4B — but a GB10 flash-attn kernel bug forces a slower attention path
that masks the speculative gain.** Google's elastic Gemma-4-E4B (unsloth Q4_K_M base) + the official
MTP drafter, via llama.cpp `--spec-type draft-mtp`.

- **GB10 flash-attn gotcha (the real story here):** with `-fa on` *or* the default `auto`, this E4B +
  MTP combination **crashes the CUDA flash-attention kernel** at load
  (`/app/ggml/src/ggml-cuda/fattn.cu:110: fatal error` in `ggml_cuda_flash_attn_ext`). Only **`-fa off`**
  loads. The E-series (MatFormer/elastic) attention config, or the MTP draft head's, isn't handled by
  this build's GB10 flash-attn kernel — note the dense **12B and 31B MTP runs use `-fa on` without
  issue**, so it's specific to the E4B path.
- **Workload:** ShareGPT V3, concurrency 32. **980/1000, 20 errors** in **883 s** — no time cap.
- **Throughput (aggregate, conc 32):** prefill **209.8 tok/s**, decode **276.6 tok/s**. TTFT median
  **1.3 s**, TPOT median **105 ms**.
- **Read this as a confounded comparison, not "MTP made it slower."** The plain E4B Q4_K_M base ran at
  **435 decode with flash attention on**; this MTP config sits at **277 with flash attention forced
  off**. Disabling flash attention alone costs more decode than the MTP drafter recovers at conc 32
  (where, per the 12B/31B results, MTP's batched gain is already small). So the drop from 435 → 277 is
  **mostly the lost flash-attention kernel, not the drafter**. An apples-to-apples MTP delta would need
  a `-fa off` base run; the takeaway recorded here is the **kernel limitation** + the standing
  conc-32-masks-spec-decode caveat, both of which matter for anyone deploying the E4B drafter on GB10.
- **Memory: 17.7 GB** — the E-series stays light (≈ the base E4B's 16.8 GB; the 98 MB draft head is
  negligible), unlike the dense Gemmas' global-attention KV cliff.
