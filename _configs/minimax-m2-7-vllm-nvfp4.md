---
title: MiniMax-M2.7 · vLLM · NVFP4
model: MiniMaxAI/MiniMax-M2.7
company: MiniMax
family: MiniMax-M2
params: ~230B (≈10B active, MoE)
engine: vLLM
quant: NVFP4
quant_rationale: saricles/MiniMax-M2.7-NVFP4-GB10 — an NVFP4 quant explicitly tagged for the GB10. Requested for queueing, but it is 130 GB and from an individual uploader — see status.
source_repo: saricles/MiniMax-M2.7-NVFP4-GB10
download_url: https://huggingface.co/saricles/MiniMax-M2.7-NVFP4-GB10
context: 65536
modalities: [text]
concurrency: 32
tags: [minimax-m2-7, MiniMax, MiniMax-M2, NVFP4, 130B+, conc-32]
status: blocked
---

**BLOCKED — two reasons: it doesn't fit, and the uploader isn't trusted.**

1. **Size: ~130 GB of weights > 121 GB unified ceiling.** 27 safetensors shards (~5 GB each) sum to
   **130.6 GB** (HF API, 2026-06-23). For an NVFP4 (W4) quant of a ~230B model this is on the high
   side — likely keeps attention/embeddings/router in higher precision — but regardless it exceeds the
   box. Unlike GGUF, a vLLM/SGLang NVFP4 load needs **all** weights resident and would OOM at load;
   there is no paging fallback. The `-GB10` tag notwithstanding, it does not fit a 121 GB GB10 with any
   KV pool. (If MiniMax shipped this for a larger-memory GB-class box, that's not this machine.)
2. **Trust: `saricles` is an individual uploader** (1.0k downloads, 13 likes) — per the trusted-repo
   policy (model's own org or a well-known quantizer only) this alone is a block until a trusted NVFP4
   appears. No official MiniMax or ModelOpt NVFP4 of M2.7 was found.

**No vLLM-viable NVFP4 path for MiniMax-M2.7 on this box right now** — both the size and the trust
gates fail. Revisit if (a) an official/ModelOpt NVFP4 under ~115 GB is published, or (b) the user
explicitly authorizes the saricles repo *and* it can be confirmed to fit (it currently cannot).

Sources: [saricles/MiniMax-M2.7-NVFP4-GB10](https://huggingface.co/saricles/MiniMax-M2.7-NVFP4-GB10).
