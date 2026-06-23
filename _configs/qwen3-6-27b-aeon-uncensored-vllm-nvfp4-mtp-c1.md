---
title: Qwen3.6-27B AEON Uncensored · vLLM · NVFP4 + MTP (XS) · conc 1
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + grafted MTP head
engine: vLLM
speculative: MTP (qwen3_5_mtp, grafted)
quant: NVFP4 (XS mixed-precision)
quant_rationale: Single-stream point of the AEON-7 NVFP4-MTP-XS sweep — same trusted native-MTP stack as the conc-32 run (stock vLLM, XS mixed-precision modelopt_fp4 + grafted Qwen3_5MTP head, no custom container). Best-case MTP latency with no batch contention; the per-stream TPOT is the headline number.
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-1]
status: done
prefill_toks: 3.74
decode_toks: 19.59
mem_gb: 109.28
mem_source: system MemAvailable delta (10s sampling) — NVFP4 (XS mixed) + grafted MTP head
spec_acceptance: mean acceptance length ~3.0 (range 2.51–3.36, median 3.03) · avg draft acceptance ~67.7% · accepted throughput ~14.7 tok/s single-stream — holds vs conc-8/conc-32
completed_at: 2026-06-23 14:24 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-1 single-stream latency point (200 prompts / 300 s cap). Trusted native-MTP path on STOCK vLLM
  # (no custom container / DFlash). Same XS mixed-precision stack as the conc-32 run.
  scripts/bench-vllm-serving.sh AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
    65536 1 200 300 256 \
    --trust-remote-code --quantization modelopt \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
  # 23/200 prompts (single stream + cold load hit the 300 s cap — expected; per-stream latency is the point).
  # 0 errors. ready after 413 s. TTFT median 281 ms, TPOT median 44.7 ms.
---

**conc-1 (single-stream) point of the AEON-7 NVFP4-MTP-XS sweep.** Best-case MTP latency — no batch
contention. Pairs with the conc-32 done run (decode 303 tok/s agg) and the conc-8 sibling. Trusted
native-MTP path on stock vLLM — no custom container.

- **Result (conc 1):** prefill 3.74 / decode **19.59** tok/s single-stream aggregate; 23/200 prompts (hit
  300 s cap — single stream + cold load, expected); peak mem 109.28 GB; **TTFT median 281 ms, TPOT median
  44.7 ms** — the headline low-conc number. TPOT is *lower* than conc-8 (52.2 ms): with one stream there's
  no batch contention, so each token decodes faster.
- **Acceptance holds:** mean accept-len **~3.0** (median 3.03), avg draft acceptance **~67.7%**, accepted
  throughput ~14.7 tok/s — identical to conc-8/conc-32. **Acceptance is workload-driven, not
  concurrency-driven** across the full AEON sweep (conc-1/8/32), exactly as on the stock Qwen3.6 sweeps.
- **vs stock:** single-stream decode 19.6 vs stock Qwen3.6-27B NVFP4+MTP conc-1 (16.85) — again modestly
  above, consistent with conc-8/conc-32. Same caveat: different model + XS recipe, not a clean comparison.
