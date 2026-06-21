---
title: SmolLM3-3B · llama.cpp · Q4_K_M
model: HuggingFaceTB/SmolLM3-3B
company: Hugging Face
family: SmolLM
params: 3B
engine: llama.cpp
quant: Q4_K_M
context: 65536
modalities: [text]
mm_served: true
tags: [Hugging Face, SmolLM, Q4_K_M, ≤4B]

status: done
prefill_toks: 7214.50
decode_toks: 105.67
mem_gb: 2.94
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-21
completed_at: 2026-06-21 19:58 +08
run_command: |
  docker run --rm --gpus all -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --bench -m /models/SmolLM3-Q4_K_M.gguf -p 512 -n 128 -ngl 99
---

Smoke-test configuration — the first end-to-end run, used to validate the manual
benchmark workflow before the larger models.

- GGUF: `ggml-org/SmolLM3-3B-GGUF` → `SmolLM3-Q4_K_M.gguf` (1.78 GiB).
- Backend: CUDA on the GB10; all 99 layers offloaded (`-ngl 99`). llama.cpp build `063d9c156 (9744)`.
- `pp512` = prefill, `tg128` = decode (llama-bench means).
- Memory via system `MemAvailable` delta — `nvidia-smi` reports no GPU memory on the GB10,
  and `docker stats` undercounted (~0.6 GiB) because CUDA's unified allocations skip the cgroup.
