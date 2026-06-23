---
title: Qwen3.6-27B · vLLM · NVFP4 + MTP
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4) + the checkpoint's own MTP module — "this checkpoint includes the MTP module, so it can act as its own speculative draft" (model card). Native multi-token-prediction spec-decode, no separate draft.
source_repo: unsloth/Qwen3.6-27B-NVFP4
download_url: https://huggingface.co/unsloth/Qwen3.6-27B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:                 # capture avg draft acceptance + mean acceptance length; cross-check vs FP8+MTP (~70%)
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). NVFP4 base + native MTP via vLLM --speculative-config (card's MTP form).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:nightly-aarch64 unsloth/Qwen3.6-27B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --trust-remote-code --dtype bfloat16 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model unsloth/Qwen3.6-27B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Queued — Qwen3.6-27B NVFP4 + native MTP on vLLM.** Spec-decode counterpart of the NVFP4 base run;
records decode-tok/s speedup vs that base plus the draft acceptance rate.

- **MTP:** `--speculative-config '{"method":"mtp","num_speculative_tokens":3}'` (card's form). The MTP
  weights ship in-repo — no separate drafter.
- **Acceptance:** capture avg draft acceptance + mean acceptance length and **cross-check against the
  FP8 + MTP runs** (measured ~67–70% / mean ~3.1 here on ShareGPT) — a big gap from FP8+MTP would flag
  a quant/format interaction with the MTP head (see BENCHMARKING.md spec-decode rules).
- **Pair:** base `qwen3-6-27b-nvfp4-vllm` + this (MTP). SGLang siblings under `*-sglang*`.
- **Repo choice — unsloth (no NVIDIA option).** Policy prefers an official `nvidia/` NVFP4 when one
  exists, but NVIDIA publishes none for the 27B (only the 35B-A3B sibling); 27B has only community NVFP4
  quants. Using **unsloth** (trusted quantizer per policy).
