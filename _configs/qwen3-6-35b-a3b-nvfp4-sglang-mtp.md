---
title: Qwen3.6-35B-A3B · SGLang · NVFP4 + MTP
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: SGLang
speculative: MTP (NEXTN)
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) + its in-repo MTP module via SGLang's NEXTN path. Preferred over unsloth per policy. NOTE — nvidia documents only a vLLM path, so SGLang support for the ModelOpt NVFP4 format must be verified at run time.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
completed_at:
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed
run_command: |
  # UNBLOCKED via the SGLang nightly (transformers 5.8.1). SGLang NEXTN/MTP. Still to verify: whether
  # SGLang's compressed-tensors path accepts nvidia ModelOpt NVFP4 (the 27B unsloth NVFP4 loaded; modelopt
  # may differ) AND whether NEXTN picks up the in-repo MTP layers. If either fails, BLOCK.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
    scripts/bench-sglang-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --speculative-algo NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
---

**Queued (UNBLOCKED via SGLang nightly) — Qwen3.6-35B-A3B NVFP4 + MTP on SGLang (NEXTN).** Completes the
MoE NVFP4 grid: `{vLLM, SGLang} × {base, MTP}`. The `nightly-dev-cu13-20260623` image loads the qwen3.6
arch the `spark` image couldn't.

- **Repo — NVIDIA official:** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4).
- **MTP via NEXTN:** `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4`. **Same ModelOpt-NVFP4-on-SGLang risk** as the base sibling — verify
  the format loads first; BLOCK if SGLang rejects it.
- **Acceptance:** capture SGLang accept length, cross-check vs the vLLM MTP run (same draft, different engine).
- **conc-8/conc-1 variants** to be added when this spec config is benchmarked.
