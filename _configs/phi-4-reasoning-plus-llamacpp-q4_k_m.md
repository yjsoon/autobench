---
title: Phi-4-reasoning-plus · llama.cpp · Q4_K_M
model: microsoft/Phi-4-reasoning-plus
company: Microsoft
family: Phi
params: 14B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer — Microsoft publishes no GGUF). Widest llama.cpp coverage.
source_repo: bartowski/microsoft_Phi-4-reasoning-plus-GGUF
download_url: https://huggingface.co/bartowski/microsoft_Phi-4-reasoning-plus-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [phi-4-reasoning-plus, Microsoft, Phi, Q4_K_M, 5-15B, conc-32]
status: done
prefill_toks: 361.25
decode_toks: 230.99
mem_gb: 31.84
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 05:16 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/microsoft_Phi-4-reasoning-plus-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model microsoft_Phi-4-reasoning-plus-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The 14B reasoning Phi — closes out the 5-15B set, and hits the time cap.** Microsoft's
Phi-4-reasoning-plus, Q4_K_M from bartowski (Microsoft ships no GGUF). The full-size sibling of the
fast Phi-4-mini-reasoning (3.8B) benchmarked earlier in the ≤4B set.

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **901/1000, 23 errors**
  (slot-split).
- **Throughput (aggregate, conc 32):** prefill **361.3 tok/s**, decode **231.0 tok/s**. TTFT median
  **897 ms**, TPOT median **118.4 ms** (≈8.4 tok/s/stream), req throughput 0.93/s — the slowest decode
  of the 14B group, consistent with the heavy reasoning workload pushing it into the time cap.
- **Read the high prefill (361) as work, not speed.** This run logged **350.7 k prompt tokens over 901
  reqs ≈ 389 tok/prompt** — the *highest* effective prompt length of any model on the same ShareGPT
  inputs, because Phi-4-reasoning's chat template injects a long reasoning system prompt. That extra
  prefill work inflates the prefill tok/s and pushes the most prompts over the 2048-tok slot limit
  (23 errors). Decode (231) is the clean cross-model figure; it's genuinely the slowest 14B here,
  since the model also emits long reasoning traces (224 k completion tokens, ~249/req near the cap).
- **Memory: 31.8 GB** — a conventional dense-attention KV at 64K ctx, right beside the DeepSeek-Distill-14B
  (30.9) and well under the Gemma-12B cliff (41.3).
- **Note:** Phi-4 is the current generation (no Phi-5). Native ctx is 32768; benchmarked at 65536 for
  cross-config comparability (llama.cpp extends via RoPE scaling), which does not affect the throughput
  comparison.
- **Slot-split errors (23):** `-c 65536 --parallel 32` → 2048 tok/slot; amplified by the long system
  prompt. Engine-config artifact, consistent across all llama.cpp runs.
