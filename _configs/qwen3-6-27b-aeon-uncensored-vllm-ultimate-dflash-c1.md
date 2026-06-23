---
title: Qwen3.6-27B AEON Uncensored · vLLM-ultimate (custom) · NVFP4 + DFlash · conc 1
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + DFlash external drafter
engine: vLLM (aeon-vllm-ultimate custom container)
speculative: DFlash (z-lab drafter, num_speculative_tokens 12)
quant: NVFP4 (XS mixed-precision)
quant_rationale: Same AEON NVFP4-XS model as the native-MTP configs, but served via the card's "DGX Spark production" recipe — the custom third-party container ghcr.io/aeon-7/aeon-vllm-ultimate:latest (vLLM 0.23.0) + the external z-lab/Qwen3.6-27B-DFlash drafter mounted at /drafter, DFlash spec-decode num_speculative_tokens=12. Run at the user's explicit request, reversing the earlier "untrusted container declined" call. SAFETY — image is untrusted; run with NO credentials (HF_TOKEN withheld), both models mounted READ-ONLY.
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-1]
status: done
prefill_toks: 38.69
decode_toks: 29.87
mem_gb: 106.96
mem_source: system MemAvailable delta (10s sampling) — custom vLLM static KV reservation (util 0.85)
spec_acceptance: 18-40% avg draft acceptance · mean acceptance length ~3.0-5.8 · per-position front-loaded across 12 draft tokens
measured_on: 2026-06-24
completed_at: 2026-06-24 00:47 +0800
engine_image: ghcr.io/aeon-7/aeon-vllm-ultimate:latest@sha256:be9e05a11da6e72607ab6f3e960993b253b673af0727005122a3266129a518e3
run_command: |
  # UNTRUSTED third-party container — NO creds, model repo root + drafter mounted READ-ONLY.
  # The cached HF snapshot itself contains symlinks into ../blobs, so mount the full repo root,
  # not only the snapshot directory.
  scripts/bench-aeon-ultimate-serving.sh 65536 1 1000 900 256
  # 106 prompts completed before the 900 s cap (0 errors). TTFT median 8111.8 ms.
---

**DONE — the custom AEON DFlash path is dramatically slower than expected even at conc-1.**

- **Result (conc 1):** prefill **38.69** tok/s, decode **29.87** tok/s aggregate; only **106 prompts**
  completed before the **900 s** cap, with **0 errors**. Median **TTFT 8111.8 ms**. This is not even
  close to the card's claimed DGX Spark production behavior.
- **This is a real performance failure, not a launch failure:** after fixing the local mount shape
  (the cached snapshot contains symlinks into Hugging Face `blobs/`, so the full repo root must be
  mounted), the custom image loaded the base as **`Qwen3_5ForConditionalGeneration`** and the drafter as
  **`DFlashDraftModel`** on its own `v0.23.0+aeon.sm121a.dflash` fork. The benchmark then ran stably to
  the time cap with no request errors.
- **Acceptance was not the main blocker:** DFlash logged **mean acceptance length ~3.0-5.8**, but the
  **avg draft acceptance rate only sat around ~18-40%** across windows because the 12-token draft has
  a very front-loaded per-position survival curve. The first few draft positions accept well, later
  ones collapse quickly. That is enough to explain some underperformance, but not the sheer magnitude of
  this slowdown.
- **Bottom line:** on this box and workload, the AEON custom container + external DFlash drafter is
  **not competitive** with the already-done stock-vLLM native-MTP path. This single-stream point is
  bad enough that the higher-concurrency points are likely measuring how quickly it degrades from an
  already poor baseline, not uncovering a hidden win.
