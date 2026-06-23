---
title: Qwen3.6-27B AEON Uncensored · vLLM · NVFP4 + MTP (XS) · conc 8
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + grafted MTP head
engine: vLLM
speculative: MTP (qwen3_5_mtp, grafted)
quant: NVFP4 (XS mixed-precision)
quant_rationale: Concurrency-8 point of the AEON-7 NVFP4-MTP-XS sweep — same trusted native-MTP stack as the conc-32 run (stock vLLM, XS mixed-precision modelopt_fp4 + grafted Qwen3_5MTP head, no custom container). Lower-batch latency characterization; acceptance should hold ~constant vs conc-32 (workload-driven).
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 8
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-8]
status: done
prefill_toks: 132.2
decode_toks: 127.52
mem_gb: 108.45
mem_source: system MemAvailable delta (10s sampling) — NVFP4 (XS mixed) + grafted MTP head
spec_acceptance: mean acceptance length ~3.0 (range 2.77–3.27, median 3.02) · avg draft acceptance ~67.5% — holds vs conc-32
completed_at: 2026-06-23 14:11 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-8 latency point (500 prompts / 300 s cap). Trusted native-MTP path on STOCK vLLM (no custom
  # container / DFlash). Same XS mixed-precision stack as the conc-32 run.
  scripts/bench-vllm-serving.sh AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
    65536 8 500 300 256 \
    --trust-remote-code --quantization modelopt \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
  # 156/500 prompts (hit 300 s cap), 0 errors. ready after 422 s. TTFT median 505 ms, TPOT median 52.2 ms.
---

**conc-8 point of the AEON-7 NVFP4-MTP-XS sweep.** Lower-batch latency characterization (cap 500 prompts /
300 s — reached 156). Pairs with the conc-32 done run (decode 303 tok/s agg) and the conc-1 sibling.
Trusted native-MTP path on stock vLLM — no custom container.

- **Result (conc 8):** prefill 132.2 / decode **127.52** tok/s aggregate; peak mem 108.45 GB; TTFT median
  505 ms, **TPOT median 52.2 ms** (valid — no reasoning-parser). Lower aggregate than conc-32 (303) purely
  because 8 streams push fewer total tokens; the per-stream TPOT is the meaningful low-conc signal.
- **Acceptance holds:** mean accept-len **~3.0** (median 3.02), avg draft acceptance **~67.5%** — identical
  to conc-32 (~3.0/~66%). **Workload-driven, not concurrency-driven**, same as the rest of the sweep.
- **vs stock:** decode 127.5 vs stock Qwen3.6-27B NVFP4+MTP conc-8 (109.05) = **~17% above** — consistent
  with the conc-32 gap (~11%). Same caveat: different model + XS recipe, not a clean comparison.
