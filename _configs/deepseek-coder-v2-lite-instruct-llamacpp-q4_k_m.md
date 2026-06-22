---
title: DeepSeek-Coder-V2-Lite-Instruct · llama.cpp · Q4_K_M
model: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
company: DeepSeek
family: DeepSeek
params: 15.7B / 2.4B (MoE)
engine: llama.cpp
quant: Q4_K_M
quant_rationale: GGUF Q4_K_M from bartowski (trusted quantizer) — widest llama.cpp coverage, strong size/quality balance.
source_repo: bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF
download_url: https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [deepseek-coder-v2-lite, DeepSeek, Q4_K_M, 16-40B]
status: done
prefill_toks: 112.01
decode_toks: 130.53
mem_gb: 38.10
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 05:43 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model DeepSeek-Coder-V2-Lite-Instruct-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The stub predicted "2.4B active → very fast." It wasn't — and that's the interesting result.**
DeepSeek's classic coding MoE (15.7B total, 2.4B active), Q4_K_M from bartowski. First of the 16-40B
bucket.

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **566/1000, 15 errors**
  (slot-split).
- **Throughput (aggregate, conc 32):** prefill **112.0 tok/s**, decode **130.5 tok/s**, TTFT median
  **588 ms**, TPOT median **248.3 ms** (≈4 tok/s/stream) — the **slowest decode of any model since the
  anomalous Nemotron-Nano-4B**, despite only 2.4B active parameters.
- **Why a 2.4B-active MoE decodes this slowly: engine × architecture, not active-param count.** Two
  things stack up on llama.cpp/GB10: (1) **DeepSeek-V2 uses MLA** (multi-head latent attention) — the
  same architecture family whose Triton decode kernel had to be disabled for the Mistral-119B vLLM run;
  llama.cpp's MLA path here is not fast. (2) It's a **fine-grained 64-expert MoE** (6 routed + 2 shared
  active); llama.cpp's expert gather/scatter at conc 32 doesn't approach the throughput vLLM's fused
  MoE kernels get on the NVFP4 Nemotron MoEs (which hit 353–389 decode at the *same* ~3B active). The
  lesson: **"small active params ⇒ fast" only holds when the engine has good kernels for that specific
  MoE+attention combo.** A vLLM re-run would be the natural follow-up to separate engine from model.
- **Memory: 38.1 GB — high for a 16B.** All 15.7B of experts must be resident (only the *compute* is
  sparse, not the footprint), plus llama.cpp materializing the MLA/KV at 64K ctx. Sparse-active does
  not mean small-resident.
- **Slot-split errors (15):** `-c 65536 --parallel 32` → 2048 tok/slot; longer ShareGPT prompts 400.
  Engine-config artifact, consistent across all llama.cpp runs.
