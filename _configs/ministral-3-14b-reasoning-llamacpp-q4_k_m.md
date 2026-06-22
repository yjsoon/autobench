---
title: Ministral-3-14B-Reasoning · llama.cpp · Q4_K_M
model: mistralai/Ministral-3-14B-Reasoning-2512
company: Mistral AI
family: Ministral
params: 14B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer — Mistral ships native weights, not GGUF). Widest llama.cpp coverage.
source_repo: bartowski/mistralai_Ministral-3-14B-Reasoning-2512-GGUF
download_url: https://huggingface.co/bartowski/mistralai_Ministral-3-14B-Reasoning-2512-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [ministral-3-14b-reasoning, Mistral AI, Ministral, Q4_K_M, 5-15B]
status: done
prefill_toks: 304.9
decode_toks: 266.03
mem_gb: 28.33
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 05:00 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/mistralai_Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model mistralai_Ministral-3-14B-Reasoning-2512-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Mistral's small reasoner — fastest decode of the 14B trio, with the tekken-tokenizer prompt
inflation showing again.** Q4_K_M from bartowski (Mistral ships native weights, not GGUF).

- **Workload:** ShareGPT V3, concurrency 32. **976/1000, 24 errors** (slot-split) in **932 s** — did
  not trip `hit_time_cap` (all prompts dispatched before the 900 s deadline; in-flight requests drained
  just past it).
- **Throughput (aggregate, conc 32):** prefill **304.9 tok/s**, decode **266.0 tok/s** — the best decode
  of the three 14Bs (vs DeepSeek-Distill 244), and a notably high prefill for the size. TTFT median
  **871 ms**, TPOT median **102.8 ms** (≈10 tok/s/stream), req throughput 1.05/s.
- **Read the high prefill with the tekken caveat — same effect as Ministral-3-3B.** This run logged
  **284 k prompt tokens over 976 reqs ≈ 291 tok/prompt**, vs ~175–205 for the non-Mistral models on the
  *same* ShareGPT inputs (~1.6×). Mistral's tekken tokenizer + chat template segment the identical text
  into more tokens, so the 305 prefill tok/s reflects **more actual prefill work per request**, not a
  faster kernel — and it also drives the **highest error count of the 14Bs (24)**, since more prompts
  cross the 2048-tok slot limit. Decode (266) is the clean cross-model number.
- **Memory: 28.3 GB** — lighter than the DeepSeek-Distill-14B (30.9) and far under the Gemma-12B cliff
  (41.3); a conventional dense-attention KV at 64K ctx.
- **Slot-split errors (24):** `-c 65536 --parallel 32` → 2048 tok/slot; amplified here by the ~1.6×
  longer effective prompts. Engine-config artifact, consistent across all llama.cpp runs.
