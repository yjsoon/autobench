---
title: Qwen3.6-35B-A3B · SGLang · NVFP4
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: SGLang
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) on SGLang — preferred over unsloth per policy (use the nvidia image when one exists). Base (non-speculative); MTP in the -mtp sibling. Cross-engine compare vs the vLLM NVFP4 run. NOTE — nvidia documents only a vLLM path, so SGLang support for the ModelOpt NVFP4 format must be verified at run time.
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
completed_at:
engine_image: lmsysorg/sglang:spark
run_command: |
  # INTENDED (not yet run). SGLang base (no spec). nvidia's card gives NO SGLang command — verify SGLang
  # loads the ModelOpt NVFP4 format (it may need an explicit --quantization modelopt equivalent, or may
  # reject it). If SGLang can't serve this format, record it and BLOCK (don't guess).
  docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 30000 \
    --context-length 65536 --trust-remote-code
  python3 scripts/bench-serving.py --base-url http://localhost:30000 --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Queued — Qwen3.6-35B-A3B NVFP4 base on SGLang (NVIDIA official quant).** Cross-engine partner to the
vLLM NVFP4 MoE base run.

- **Repo — NVIDIA official:** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
  (ModelOpt v0.44.0), per the new policy (prefer nvidia where it exists).
- **Risk to verify:** NVIDIA documents **only a vLLM** path for this checkpoint. SGLang's support for the
  ModelOpt NVFP4 format on GB10 is unconfirmed — if SGLang can't load it, record the error and **BLOCK**
  rather than guess. (If SGLang only supports a different NVFP4 packing, the unsloth quant may be the
  SGLang-viable fallback — note it then.)
- **Spec-decode:** in `qwen3-6-35b-a3b-nvfp4-sglang-mtp`. This is the non-spec baseline.
