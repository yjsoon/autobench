---
title: MiniMax-M2.7 · llama.cpp · UD-IQ4_XS
model: MiniMaxAI/MiniMax-M2.7
company: MiniMax
family: MiniMax-M2
params: ~230B (≈10B active, MoE)
engine: llama.cpp
quant: UD-IQ4_XS
quant_rationale: unsloth dynamic IQ4_XS GGUF — the largest ≈4-bit MiniMax-M2.7 quant that FITS the 121 GB GB10 (108.4 GB weights, ~12 GB headroom for KV). The requested MXFP4_MOE (136 GB) and saricles NVFP4 (130 GB) both exceed the box — see those configs. This is the runnable 4-bit stand-in from the same trusted unsloth repo.
source_repo: unsloth/MiniMax-M2.7-GGUF
download_url: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF/tree/main/UD-IQ4_XS
context: 32768
modalities: [text]
concurrency: 32
tags: [minimax-m2-7, MiniMax, MiniMax-M2, IQ4_XS, 130B+, conc-32]
status: pending
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda — 4-shard GGUF auto-loaded from shard 1.
  # Context reduced to 32768 (vs 65536) to leave KV room: 108 GB weights on a 121 GB box.
  ./scripts/bench-llamacpp-serving.sh \
    minimax-m2-7/UD-IQ4_XS/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
    32768 32 1000 900 256 99
---

**Queued — the runnable MiniMax-M2.7 4-bit on llama.cpp.** The user's first two MiniMax requests
(MXFP4_MOE 136 GB, saricles NVFP4 130 GB) both exceed the 121 GB unified ceiling
(see `minimax-m2-7-llamacpp-mxfp4` / `minimax-m2-7-vllm-nvfp4`). `UD-IQ4_XS` is the largest ≈4-bit
tier that fits: **108.4 GB** across 4 shards (49.6 + 49.6 + 9.2 + index), leaving ~12 GB for KV/compute
— comparable to how gpt-oss-120b filled the box to ~8 GB headroom.

- **Download in progress** (108 GB → `~/models/minimax-m2-7/UD-IQ4_XS/`, overlapping the gpt-oss GPU run).
- **Context set to 32768, not 65536** — at 108 GB resident the KV pool must fit in ~12 GB headroom;
  65536 risks OOM. Start at 32768/conc-32; tune down if the load OOMs, up if headroom allows. ShareGPT
  prompts are short enough that 32768 across 32 slots is ample.
- **Pre-run check:** confirm the pinned `llama.cpp:full-cuda` build recognizes the MiniMax-M2 arch
  (unsloth's [M2.7 guide](https://unsloth.ai/docs/models/tutorials/minimax-m27) notes a recent
  llama.cpp is required) — if the load rejects the arch, pull a newer `full-cuda` tag and record the
  digest. MiniMax-M2 is **not** harmony, so it should serve cleanly on llama-server's chat endpoint
  (unlike gpt-oss).

<!-- results pending — runs after the gpt-oss-120b SGLang+EAGLE3 sweep frees the GPU -->
