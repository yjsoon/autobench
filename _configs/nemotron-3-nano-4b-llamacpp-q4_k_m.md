---
title: Nemotron-3 Nano-4B · llama.cpp · Q4_K_M
model: nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
company: NVIDIA
family: Nemotron
params: 4B (dense)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from NVIDIA's own GGUF repo — widest llama.cpp coverage, strong size/quality balance.
source_repo: nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
download_url: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-4B-GGUF
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [nvidia-nemotron-3-nano-4b, NVIDIA, Nemotron, Q4_K_M, ≤4B, Spark recipe, conc-32]
status: done
prefill_toks: 112.44
decode_toks: 141.32
mem_gb: 21.64
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 03:10 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  # ctx 65536 with --parallel 32 = 2048 tok/slot (see slot-split note).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**First of the small→big llama.cpp sweep — and, in hindsight, the slow outlier of the ≤4B set.**
NVIDIA's own Q4_K_M GGUF (2.7 GB file).

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **545/1000 prompts** with
  **11 errors** (the long-prompt slot-split artifact below). Loaded in **16 s**.
- **Throughput (aggregate, conc 32):** prefill **112.4 tok/s**, decode **141.3 tok/s**, TPOT median
  **211 ms** (≈4.7 tok/s/stream).
- **This is anomalously slow for the size — and it's the model, NOT llama.cpp.** My first read was
  "llama.cpp scales poorly at conc 32," but the *very next two* ≤4B runs on the identical engine /
  quant / ctx disproved that: **Granite-4.1-3B hit 617 tok/s and Phi-4-mini-reasoning 552 tok/s**
  decode (both finished without the time-cap). So llama.cpp parallelizes fine here — Nemotron-Nano-4B
  specifically decodes ~4× slower per token (TPOT 211 ms vs Phi's 50 ms). The cause is this model's
  architecture/GGUF on llama.cpp, not the engine's concurrency. Worth a follow-up: re-run it on
  vLLM/SGLang to see whether the slowness is llama.cpp-specific or intrinsic to the model. It's also
  a reasoning model (132 k output tokens over 545 reqs ≈ 243 each), which compounds the time-cap.
- **Slot-split errors (11):** `-c 65536 --parallel 32` gives each slot only **2048 tokens** of
  context; ShareGPT prompts longer than that return HTTP 400. Same artifact seen on the SmolLM3
  shakedown — an engine-config consequence, not a model failure. Kept ctx 65536 for comparability
  across the llama.cpp runs.
- **Memory:** **21.6 GB** MemAvailable delta — dominated not by the 2.7 GB weights but by the
  **full 65536-token KV cache llama.cpp pre-allocates** at load (plus CUDA context). Small model,
  but the big static context reservation makes the footprint look large.
