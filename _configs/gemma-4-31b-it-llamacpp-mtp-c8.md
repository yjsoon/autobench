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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
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

**Queued — concurrency-8 variant of [Gemma 4 31B · llama.cpp · Q4_K_M + MTP].** Low-concurrency point for the spec-decode concurrency sweep (the run is otherwise identical; cap reduced to 500 prompts / 300 s since low-conc runs are latency characterizations, not throughput-to-1000). Compare decode + TPOT against this config's conc-32 run to see how the speculative gain scales as the batch empties.
