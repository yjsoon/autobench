---
title: MiniMax-M2.7 · llama.cpp · MXFP4_MOE
model: MiniMaxAI/MiniMax-M2.7
company: MiniMax
family: MiniMax-M2
params: ~230B (≈10B active, MoE)
engine: llama.cpp
quant: MXFP4_MOE
quant_rationale: unsloth's MXFP4_MOE GGUF (gpt-oss-style FP4 MoE quant, fast-at-slight-quality-cost). Requested explicitly, but the single MXFP4_MOE tier is 136.2 GB — see status.
source_repo: unsloth/MiniMax-M2.7-GGUF
download_url: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF/tree/main/MXFP4_MOE
context: 65536
modalities: [text]
concurrency: 32
tags: [minimax-m2-7, MiniMax, MiniMax-M2, MXFP4, 130B+, conc-32]
status: blocked
---

**BLOCKED — does not fit. The MXFP4_MOE GGUF is 136.2 GB; the GB10 has a 121 GB unified-memory
ceiling.** Weights alone exceed total memory by ~15 GB, before any KV cache, activations, or OS
overhead. On this box CPU and GPU share **one** unified pool — there is no separate system RAM to
offload MoE experts into, so the only way to "run" 136 GB would be llama.cpp disk-mmap paging from
the NVMe. Since MiniMax-M2 is a sparse MoE whose active experts change every token, that path is
fully disk-bound and would not produce a meaningful throughput number — so it is blocked rather than
queued (and avoids a multi-hour 136 GB download for a guaranteed OOM).

- **Sizes in `unsloth/MiniMax-M2.7-GGUF` (verified via HF API, 2026-06-23):** MXFP4_MOE **136.2 GB**
  (4 shards, single tier — no smaller MXFP4). For reference the unsloth docs quote ~124 GB for the
  earlier M2 MXFP4_MOE; the M2.7 tier is larger.
- **Fitting llama.cpp alternatives in the same repo** (if a MiniMax-M2.7 speed number on llama.cpp is
  wanted, these land near gpt-oss-120b's ~105 GB resident with headroom for KV):
  `UD-Q3_K_XL` **101.9 GB** · `UD-IQ4_XS` **108.4 GB** · `UD-IQ4_NL` **110.8 GB**. `UD-Q4_K_*`
  (131–141 GB) and up do **not** fit. These are unsloth dynamic quants (trusted repo) — a `UD-IQ4_XS`
  config would be the runnable stand-in for "4-bit MiniMax-M2.7 on llama.cpp."
- **Secondary note (moot while blocked on size):** MiniMax-M2 arch needs a recent llama.cpp build
  ([unsloth M2.7 guide](https://unsloth.ai/docs/models/tutorials/minimax-m27)) — verify the pinned
  `ghcr.io/ggml-org/llama.cpp:full-cuda` digest supports it before running any fitting quant.

Sources: [unsloth/MiniMax-M2.7-GGUF](https://huggingface.co/unsloth/MiniMax-M2.7-GGUF) ·
[unsloth M2.7 run guide](https://unsloth.ai/docs/models/tutorials/minimax-m27).
