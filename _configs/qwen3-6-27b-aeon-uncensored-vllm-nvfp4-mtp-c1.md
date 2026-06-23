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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-1 single-stream latency point (200 prompts / 300 s cap). Trusted native-MTP path on STOCK vLLM
  # (no custom container / DFlash). Same XS mixed-precision stack as the conc-32 run.
  scripts/bench-vllm-serving.sh AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS \
    65536 1 200 300 256 \
    --trust-remote-code --quantization modelopt \
    --speculative-config '{"method":"qwen3_5_mtp","num_speculative_tokens":3}'
---

**conc-1 (single-stream) point of the AEON-7 NVFP4-MTP-XS sweep.** Best-case MTP latency — no batch
contention. Pairs with the conc-32 done run (decode 303 tok/s agg) and the conc-8 sibling. Trusted
native-MTP path on stock vLLM — no custom container.

- **What to capture:** single-stream decode tok/s + TPOT (the meaningful low-conc metric), peak mem, MTP
  accept-len. Acceptance is expected to hold ~3.0 (workload-driven across the whole Qwen3.6 sweep).
