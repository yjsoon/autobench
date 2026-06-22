---
title: Devstral-Small 24B · llama.cpp · Q4_K_M
model: mistralai/Devstral-Small-2507
company: Mistral AI
family: Devstral
params: 24B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer — Mistral ships native weights). Widest llama.cpp coverage.
source_repo: bartowski/mistralai_Devstral-Small-2507-GGUF
download_url: https://huggingface.co/bartowski/mistralai_Devstral-Small-2507-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [devstral-small, Mistral AI, Devstral, Q4_K_M, 16-40B]
status: done
prefill_toks: 967.85
decode_toks: 139.0
mem_gb: 36.62
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 06:21 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/mistralai_Devstral-Small-2507-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model mistralai_Devstral-Small-2507-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The agentic coder whose giant system prompt dominates the run — a prefill-bound profile unlike any
other model here.** Mistral's Devstral-Small (24B dense), Q4_K_M from bartowski.

- **Repo note:** the stub's `Devstral-Small-2-24B-Instruct-2512` does not exist on HF; the current
  real 24B Devstral is **Devstral-Small-2507**, benchmarked here (a model-list-name recovery, per the
  "verify exact repo at download time" policy).
- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **705/1000, 82 errors**
  (slot-split — the most of any non-Mistral run).
- **Throughput (aggregate, conc 32):** prefill **967.9 tok/s** (2nd-highest measured, behind only the
  Ministral-3-3B), decode **139.0 tok/s** (very low for a 24B), TTFT median **579 ms**, TPOT median
  **195 ms**.
- **Why the lopsided profile: ~1360 prompt tokens *per request*.** This run logged **958 k prompt
  tokens over 705 requests** — roughly **1360 tok/prompt**, ~7× the ~185 of the plain small models on
  the *same* ShareGPT inputs and ~4.7× even the other Mistral (tekken) models. Devstral is an **agentic
  coding model whose chat template injects a large tool-use/system preamble**, so every request carries
  a huge prefix. The consequences cascade: (1) prefill tok/s looks enormous (968) because there's
  genuinely that much prefill work; (2) **decode is starved** — the engine spends most of each step on
  prefill, so output throughput collapses to 139 and the run **hits the time cap** at 705; (3) the
  **82 slot-split errors** follow directly, since a 1360-token prompt + output routinely blows past the
  2048-tok/slot limit. The decode number here is *not* comparable to the other 24-32B models — it's an
  artifact of the prompt-length profile, not the model's raw decode speed. Memory **36.6 GB**, normal
  for a dense 24B.
- **Slot-split errors (82):** `-c 65536 --parallel 32` → 2048 tok/slot, overwhelmed by the ~1360-token
  prompts. Engine-config artifact, amplified to its extreme here.
