---
title: Qwen3.6-27B AEON Uncensored · vLLM · NVFP4 + MTP (XS)
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + grafted MTP head
engine: vLLM
speculative: MTP (qwen3_5_mtp, grafted)
quant: NVFP4 (XS mixed-precision)
quant_rationale: AEON-7's NVFP4-MTP-XS — an abliterated/uncensored fine-tune of Qwen3.6-27B (via AEON-7/...-Uncensored-BF16), NVFP4-quantized with nvidia-modelopt 0.43.0 (NVFP4_DEFAULT_CFG). "XS" = mixed precision: GDN projection matmuls → NVFP4, linear_attn.conv1d (SSM kernel) + vision tower kept BF16 for stability (~21 GB disk, ~22 GB VRAM). MTP head grafted (15 BF16 tensors) from stock Qwen3.6-27B. Individual-uploader DERIVATIVE added at the user's explicit request (same exception as cosmicproc/shieldstar). NOT a clean comparison to stock Qwen3.6-27B — it is a different (abliterated) model.
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:                 # capture MTP accept length; the card cites ~45% DFlash accept @9k (different path)
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). TRUSTED PATH ONLY: native MTP on STOCK vLLM nightly-aarch64.
  # The card's "DGX Spark production" recipe uses a CUSTOM third-party container
  # (ghcr.io/aeon-7/aeon-vllm-ultimate) + a DFlash external drafter — DECLINED: do not run an untrusted
  # container. The repo's own native-MTP form runs on stock vLLM with --quantization modelopt:
  scripts/bench-vllm-serving.sh AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
    65536 32 1000 900 256 \
    --trust-remote-code --quantization modelopt \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
---

**Queued — AEON-7 uncensored Qwen3.6-27B, NVFP4-MTP-XS, on stock vLLM (trusted path only).** A
well-documented but **individual-uploader abliterated derivative** of Qwen3.6-27B, added at the user's
explicit request. Throughput-only benchmark (ShareGPT general chat) — the speed-only scope makes this a
benign inference-rate measurement; the model's lack of refusals is irrelevant to tok/s.

- **What it is:** abliterated ("uncensored", 0/100 refusals, KL ~0.0005 vs base) fine-tune → NVFP4 (XS
  mixed: GDN proj NVFP4, conv1d/vision BF16) → grafted MTP head. **Distinct model tag**
  (`qwen3.6-27b-aeon-uncensored`) so it does NOT pollute the clean stock-Qwen3.6-27B comparison group.
- **Trust & safety:** individual uploader (AEON-7, 41k dl/mo, 48 likes), Apache-2.0, modelopt NVFP4
  format confirmed (served via stock vLLM `--quantization modelopt`). Run per the user's explicit
  request — same exception as cosmicproc/shieldstar.
- **DECLINED — custom container + DFlash path.** The card's DGX Spark recipe pulls a third-party image
  (`ghcr.io/aeon-7/aeon-vllm-ultimate:latest`) and a `/drafter` for `method:dflash,num_speculative_tokens:12`.
  We do **not** run untrusted containers; benchmark only the **native `qwen3_5_mtp`** path on stock
  `nightly-aarch64`. (The card's quoted "340 tok/s @ c=64, 45% DFlash accept @9k" come from that custom
  path and aren't directly reproducible here.)
- **Expectation:** throughput should land near the stock unsloth NVFP4+MTP 27B run (abliteration is a
  weight tweak, not an arch change); the value here is confirming the AEON XS mixed-precision packing
  serves on stock vLLM and what its native-MTP acceptance looks like.
- **At run time:** if stock vLLM rejects the XS mixed-precision packing or the grafted MTP head, record
  the error and BLOCK rather than guess.
