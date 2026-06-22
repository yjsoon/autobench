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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
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

**Queued — concurrency-1 variant of [Gemma 4 12B · llama.cpp · Q4_K_M + MTP].** Low-concurrency point for the spec-decode concurrency sweep (the run is otherwise identical; cap reduced to 500 prompts / 300 s since low-conc runs are latency characterizations, not throughput-to-1000). Compare decode + TPOT against this config's conc-32 run to see how the speculative gain scales as the batch empties.
