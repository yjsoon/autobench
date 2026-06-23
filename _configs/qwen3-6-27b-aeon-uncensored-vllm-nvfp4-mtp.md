---
title: Qwen3.6-27B AEON Uncensored · vLLM · NVFP4 + MTP (XS)
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + grafted MTP head
engine: vLLM
speculative: MTP (qwen3_5_mtp, grafted)
quant: NVFP4 (XS mixed-precision)
quant_rationale: AEON-7's NVFP4-MTP-XS — an abliterated/uncensored fine-tune of Qwen3.6-27B (via AEON-7/...-Uncensored-BF16), NVFP4-quantized with nvidia-modelopt 0.43.0 (NVFP4_DEFAULT_CFG). "XS" = mixed precision (GDN projection matmuls → NVFP4; linear_attn.conv1d SSM kernel + vision tower kept BF16 for stability, ~21 GB disk, ~22 GB VRAM). MTP head grafted (15 BF16 tensors) from stock Qwen3.6-27B. Individual-uploader DERIVATIVE added at the user's explicit request (same exception as cosmicproc/shieldstar). NOT a clean comparison to stock Qwen3.6-27B — it is a different (abliterated) model.
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 312.31
decode_toks: 303.32
mem_gb: 108.58
mem_source: system MemAvailable delta (10s sampling) — NVFP4 (XS mixed) + grafted MTP head, KV cache 76 GiB
spec_acceptance: mean acceptance length ~3.0 (range 2.7–3.6, median ~3.0) · avg draft acceptance ~66% · per-position ~0.84/0.68/0.54 — matches stock Qwen3.6-27B MTP
completed_at: 2026-06-23 13:56 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # TRUSTED PATH ONLY: native MTP on STOCK vLLM nightly-aarch64 (no custom container). The card's
  # "DGX Spark production" recipe uses a CUSTOM third-party container (ghcr.io/aeon-7/aeon-vllm-ultimate)
  # + a DFlash external drafter — DECLINED (do not run an untrusted container). The repo's own native-MTP
  # form runs on stock vLLM with --quantization modelopt:
  scripts/bench-vllm-serving.sh AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
    65536 32 1000 900 256 \
    --trust-remote-code --quantization modelopt \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
  # 1000/1000 prompts, 0 errors, 843.9 s (did NOT hit 900 s cap — full coverage). ready after 601 s.
  # vLLM maps method qwen3_5_mtp -> mtp; draft arch resolves as Qwen3_5MTP; quantization=modelopt_fp4.
  # TTFT median 844 ms, TPOT median 94.9 ms (valid here — no reasoning-parser).
---

**DONE — AEON-7 uncensored Qwen3.6-27B, NVFP4-MTP-XS, on stock vLLM (trusted path only).** An
**individual-uploader abliterated derivative** of Qwen3.6-27B, added at the user's explicit request.
Throughput-only benchmark (ShareGPT general chat) — a benign inference-rate measurement; the model's lack
of refusals is irrelevant to tok/s. The native-MTP path serves cleanly on **stock vLLM**, no custom
container needed.

- **Result (conc 32):** prefill 312.3 / decode **303.32** tok/s aggregate; **1000/1000 prompts, 0 errors**,
  843.9 s (did **not** hit the 900 s cap → full coverage). Peak mem **108.58 GB**. TTFT median 844 ms,
  TPOT median 94.9 ms.
- **XS packing + grafted MTP head load on stock vLLM** — no rejection. vLLM resolves the arch as
  `Qwen3_5ForConditionalGeneration`, maps `method:qwen3_5_mtp → mtp` (deprecation alias), loads the draft
  as `Qwen3_5MTP`, and accepts the mixed-precision `modelopt_fp4` weights (GDN proj NVFP4 / conv1d+vision
  BF16). This **confirms the AEON XS format is serviceable on the stock engine** without the custom path.
- **MTP acceptance matches stock:** mean accept-len **~3.0** (range 2.7–3.6, median ~3.0), avg draft
  acceptance **~66%**, per-position ~0.84/0.68/0.54 — **indistinguishable from stock Qwen3.6-27B MTP**
  (~3.0/~66%). Abliteration is a weight tweak; it does not change MTP draft quality.
- **Speed vs stock:** decode **303 tok/s** lands **~11% above** the stock Qwen3.6-27B NVFP4+MTP run
  (274 tok/s). **Not a clean comparison** — different (abliterated) weights *and* a different quant recipe
  (XS mixed-precision vs the stock unsloth NVFP4 packing). Treat as "serves at least as fast as stock,"
  not a validated speedup; the distinct model tag keeps it out of the clean stock-27B group.
- **DECLINED — custom container + DFlash path.** The card's DGX Spark recipe pulls a third-party image
  (`ghcr.io/aeon-7/aeon-vllm-ultimate:latest`) + a `/drafter` for `method:dflash,num_speculative_tokens:12`.
  Untrusted containers are **not** run here; only the native `qwen3_5_mtp` path on stock `nightly-aarch64`
  was benchmarked. (The card's "340 tok/s @ c=64, 45% DFlash accept @9k" come from that custom path and
  are not reproduced here.)
- **Trust & safety:** individual uploader (AEON-7, 41k dl/mo, 48 likes), Apache-2.0, modelopt NVFP4
  confirmed. Run per the user's explicit request — same exception as cosmicproc/shieldstar.
