---
title: Ministral-3-3B · llama.cpp · Q4_K_M
model: mistralai/Ministral-3-3B-Instruct-2512
company: Mistral AI
family: Ministral
params: 3.4B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer — Mistral ships native weights, not GGUF). Widest llama.cpp coverage.
source_repo: bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF
download_url: https://huggingface.co/bartowski/mistralai_Ministral-3-3B-Instruct-2512-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [Mistral AI, Ministral, Q4_K_M, ≤4B]

status: done
prefill_toks: 1457.3
decode_toks: 523.73
mem_gb: 19.12
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 03:35 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model mistralai_Ministral-3-3B-Instruct-2512-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A fast dense 3.4B — with the highest prefill throughput in the whole benchmark.** Mistral's
Ministral-3-3B, Q4_K_M from bartowski.

- **Workload:** ShareGPT V3, concurrency 32. **956/1000, 44 errors** in **438 s** — **no time-cap**.
- **Throughput (aggregate, conc 32):** prefill **1457.3 tok/s** (highest of any config measured),
  decode **523.7 tok/s**. TTFT median **156 ms**, TPOT median **54.7 ms** (≈18 tok/s/stream). Another
  well-behaved dense small model on llama.cpp, right alongside Phi-4-mini (552) — and far above the
  anomalous Nemotron-Nano-4B (141).
- **The sky-high prefill is partly a token-count effect — read it with care.** This run's server
  reported **638 k prompt tokens** over 956 requests (~667 tok/prompt) vs ~185 tok/prompt for the
  other small models on the *same* ShareGPT inputs — roughly **3.6×**. That inflates both the prefill
  tok/s and the slot-split error count (44, the most of the ≤4B set). The likely cause is Mistral's
  chat template injecting a substantial system prompt and/or the tekken tokenizer segmenting this
  text into more tokens. So the 1457 prefill number reflects more *actual prefill work per request*,
  not a 4× faster kernel — it's genuinely processing more tokens. Decode (523.7) is the cleaner
  cross-model comparison and lands as expected for a dense 3.4B.
- **Memory:** **19.1 GB** (weights + the pre-allocated 64K KV).
- **Slot-split errors (44):** `-c 65536 --parallel 32` → 2048 tok/slot; with ~3.6× longer effective
  prompts, more cross that limit and 400. Same engine-config artifact, amplified by the prompt-length
  effect above.
