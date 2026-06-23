---
title: Gemma 4 12B · vLLM · NVFP4 + MTP · RedHatAI
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: RedHatAI/gemma-4-12B-it-NVFP4 base + Google's official `google/gemma-4-12B-it-assistant` drafter, using vLLM's native Gemma-4 Unified MTP path. vLLM's current docs explicitly route Gemma-4 assistant checkpoints through `method=mtp`, not generic draft-model speculation.
source_repo: RedHatAI/gemma-4-12B-it-NVFP4
download_url: https://huggingface.co/RedHatAI/gemma-4-12B-it-NVFP4
context: 65536
modalities: [text, image, audio, video]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, NVFP4, 5-15B, conc-32]
status: done
prefill_toks: 903.43
decode_toks: 782.37
mem_gb: 108.30
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + Gemma4 MTP head
spec_acceptance: 55-58% avg draft acceptance · mean acceptance length ~2.6-2.7 · per-position ~0.76/0.56/0.41 (num_speculative_tokens=3)
measured_on: 2026-06-23
completed_at: 2026-06-23 23:35 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # vLLM docs (June 2026) say Gemma-4 Unified assistant checkpoints use the native MTP path:
  # `--speculative-config {"method":"mtp","model":"google/gemma-4-12B-it-assistant",...}`.
  # Repo convention uses 3 speculative tokens for the throughput benchmark sweep.
  scripts/bench-vllm-serving.sh RedHatAI/gemma-4-12B-it-NVFP4 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-12B-it-assistant","num_speculative_tokens":3}'
  # 1000/1000 prompts, 0 errors, 298.3 s. TTFT median 273.2 ms, TPOT median 35.8 ms.
  # SpecDecoding mostly held mean acceptance length ~2.6-2.7 and avg draft acceptance ~55-58%,
  # with a brief late-run dip to 2.28 / 42.7%.
---

**DONE — the official Google assistant materially improves the trusted Red Hat AI NVFP4 base on the
maintained vLLM nightly.**

- **Result (conc 32):** prefill **903.43** tok/s, decode **782.37** tok/s aggregate; **1000/1000, 0
  errors** in **298.3 s**. Median **TTFT 273.2 ms**, median **TPOT 35.8 ms**, request throughput
  **3.352 req/s**.
- **Speedup vs base:** decode rises from **503.81 -> 782.37 tok/s**, a **+55%** gain on the same
  ShareGPT conc-32 sweep. Prefill also rises sharply (**582.39 -> 903.43 tok/s**), which is consistent
  with the assistant reducing queued decode pressure enough for the server to spend more time feeding
  fresh work.
- **Acceptance:** the native Gemma-4 MTP path stayed mostly in the expected band for ShareGPT:
  **mean acceptance length ~2.6-2.7**, **avg draft acceptance ~55-58%**, per-position about
  **0.76 / 0.56 / 0.41** at `num_speculative_tokens=3`. That is below the "~3.0 / ~70%+" ideal but
  still healthy enough to deliver a real throughput win. There was one brief late-run dip to
  **2.28 / 42.7%**, then recovery.
- **Load/runtime behavior:** this confirms the current **`nightly-aarch64`** image can handle the full
  `Gemma4UnifiedForConditionalGeneration` base plus the official `Gemma4MTPModel` assistant on Spark.
  Server readiness took **250 s**. vLLM again forced **TRITON_ATTN** because Gemma-4 has heterogeneous
  head dimensions, used **FlashInferCutlassNvFp4LinearKernel** for NVFP4 GEMM, and warned that the
  speculative settings reduced `max_num_scheduled_tokens` to **2496**. Even with that conservative
  scheduler budget, the run still beat the base cleanly.
- **Takeaway:** unlike the unresolved image split on some Gemma-4 NVFP4 paths, this exact **12B
  RedHatAI NVFP4 + Google assistant** combination is already a production-usable vLLM path on the
  maintained nightly.
