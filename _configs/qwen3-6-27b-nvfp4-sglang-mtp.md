---
title: Qwen3.6-27B · SGLang · NVFP4 + MTP
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: SGLang
speculative: MTP (NEXTN)
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4) + its in-repo MTP module, driven via SGLang's NEXTN speculative path (the card's SGLang spec form). Native multi-token-prediction, no separate draft.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32, Spark recipe]
status: done
prefill_toks: 197.2
decode_toks: 196.62
mem_gb: 104.01
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV + NEXTN drafter
spec_acceptance: accept len ~3.0 (range 2.85–3.22) · accept rate ~0.69 — from SGLang decode-batch logs
measured_on: 2026-06-23
completed_at: 2026-06-23 12:18 +08
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # UNBLOCKED via the SGLang nightly (transformers 5.8.1) — the spark image can't load qwen3.6.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
    scripts/bench-sglang-serving.sh unsloth/Qwen3.6-27B-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --speculative-algo NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  # 716/1000 prompts, 0 errors (hit 900 s cap, dur 931.7s). ready after 304 s.
  # TTFT median 11342 ms (deep queue at conc 32), TPOT median 110.2 ms.
---

**UNBLOCKED on the SGLang nightly — MTP completes the NVFP4 engine × spec grid.** The stock
`lmsysorg/sglang:spark` image (transformers 4.57.1) couldn't load the Qwen3.6 (`qwen3_5`) arch; the
**`nightly-dev-cu13-20260623-ba9d5aed`** image (transformers 5.8.1) loads it and runs NEXTN cleanly.

- **Result (conc 32):** prefill 197.2 / decode **196.62** tok/s aggregate; **716/1000, 0 errors** (hit the
  900 s cap). Peak mem **104.01 GB**.
- **MTP speedup vs base:** decode **196.62 (MTP) vs 177.99 (base)** on SGLang — only **+10%**. Acceptance
  is healthy (accept len ~3.0, rate ~0.69), so the modest gain is engine overhead, not a draft-quality
  problem: SGLang's hybrid scheduler runs **GDN/mamba layers** alongside the NEXTN drafter (logs show
  `mamba num`/`mamba usage ~0.66` per decode batch), and that bookkeeping eats most of the spec win at
  conc-32.
- **Cross-engine vs vLLM MTP:** decode **196.6 (SGLang) vs 274.1 (vLLM)** at the same NVFP4 / 65536 /
  conc-32 / same in-repo MTP module. **vLLM is ~40% faster with MTP** even though both see ~3.0 accept
  length — vLLM's native `qwen3_5_mtp` path extracts far more of the acceptance into throughput here.
  (Base: vLLM 187.7 vs SGLang 178.0, only ~5% — so the gap is almost entirely on the spec path.)
- **Acceptance cross-check:** SGLang accept len ~3.0 ≈ vLLM MTP mean-len 3.0 / accept 67% — the **drafts
  agree across engines**, confirming the in-repo MTP module behaves consistently; the throughput delta is
  purely engine-side.
- **Engine image:** pinned the nightly via the `SGLANG_IMAGE=` wrapper override.
