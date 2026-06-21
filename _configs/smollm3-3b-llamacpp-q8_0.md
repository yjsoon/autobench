---
title: SmolLM3-3B · llama.cpp · Q8_0
model: HuggingFaceTB/SmolLM3-3B
company: Hugging Face
family: SmolLM
params: 3B
engine: llama.cpp
quant: Q8_0
quant_rationale: Q8_0 — near-lossless reference point to quantify Q4_K_M's quality/speed tradeoff.
source_repo: ggml-org/SmolLM3-3B-GGUF
download_url: https://huggingface.co/ggml-org/SmolLM3-3B-GGUF
context: 65536
modalities: [text]
mm_served: true
tags: [Hugging Face, SmolLM, Q8_0, ≤4B]

status: done
prefill_toks: 6391.07
decode_toks: 70.61
mem_gb: 4.18
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-21
completed_at: 2026-06-21 19:59 +08
run_command: |
  docker run --rm --gpus all -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --bench -m /models/SmolLM3-Q8_0.gguf -p 512 -n 128 -ngl 99
---

Q8_0 companion to the Q4_K_M smoke test — the quant axis of the sweep.

- GGUF: `ggml-org/SmolLM3-3B-GGUF` → `SmolLM3-Q8_0.gguf` (3.04 GiB).
- vs Q4_K_M: decode is slower (70.6 vs 105.7 tok/s — Q8 moves ~2× the bytes per weight, and
  decode is memory-bandwidth bound) while prefill stays high (compute bound). Memory: 4.18 vs 2.94 GiB.
- Backend: CUDA on the GB10, `-ngl 99`. llama.cpp build `063d9c156 (9744)`.
