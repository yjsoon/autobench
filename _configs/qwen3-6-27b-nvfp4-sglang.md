---
title: Qwen3.6-27B · SGLang · NVFP4
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: SGLang
quant: NVFP4
quant_rationale: Unsloth's NVFP4 (W4A4) quant of Qwen3.6-27B (unsloth/Qwen3.6-27B-NVFP4) on SGLang — the card lists SGLang as a supported serving path. Base (non-speculative) run; the repo's MTP module is exercised in the -mtp sibling. Cross-engine compare against the vLLM NVFP4 run.
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
completed_at:
engine_image: lmsysorg/sglang:spark
run_command: |
  # INTENDED (not yet run). SGLang base (no spec) — drop the card's --speculative-* flags for the base run.
  docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path unsloth/Qwen3.6-27B-NVFP4 --host 0.0.0.0 --port 30000 \
    --context-length 65536 --trust-remote-code
  python3 scripts/bench-serving.py --base-url http://localhost:30000 --model unsloth/Qwen3.6-27B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Queued — Qwen3.6-27B NVFP4 base run on SGLang.** Cross-engine partner to the vLLM NVFP4 base run:
same model/quant/context/concurrency, different engine, to see which serves NVFP4 faster on GB10.

- **Model:** `unsloth/Qwen3.6-27B-NVFP4`, served text-only at **65536** ctx (native 262K).
- **Spec-decode:** exercised separately in `qwen3-6-27b-nvfp4-sglang-mtp` (SGLang NEXTN/MTP). This page
  is the non-spec baseline.
- **Note at run time:** confirm SGLang's `spark` image loads the `qwen3_5` multimodal arch + NVFP4
  kernels; if it rejects either, record it and (if needed) BLOCK rather than guess.
- **Repo choice — unsloth (no NVIDIA option).** Policy prefers an official `nvidia/` NVFP4 when one
  exists, but NVIDIA publishes none for the 27B (only the 35B-A3B sibling); 27B has only community NVFP4
  quants. Using **unsloth** (trusted quantizer per policy).
