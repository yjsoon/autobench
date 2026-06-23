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
tags: [qwen3.6-27b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:                 # capture SGLang accept length; cross-check vs the vLLM MTP run
completed_at:
engine_image: lmsysorg/sglang:spark
run_command: |
  # INTENDED (not yet run). SGLang NEXTN/MTP form from the model card.
  docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path unsloth/Qwen3.6-27B-NVFP4 --host 0.0.0.0 --port 30000 \
    --context-length 65536 --trust-remote-code \
    --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
  python3 scripts/bench-serving.py --base-url http://localhost:30000 --model unsloth/Qwen3.6-27B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Queued — Qwen3.6-27B NVFP4 + MTP on SGLang (NEXTN).** SGLang's speculative path for the same in-repo
MTP module that the vLLM `-mtp` sibling uses, so the four NVFP4 pages form a clean engine × spec grid.

- **MTP via NEXTN:** `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4` (card's SGLang form). NEXTN is SGLang's name for the MTP/next-N
  draft path.
- **Acceptance:** capture SGLang's reported accept length and **cross-check against the vLLM MTP run**
  (same draft, different engine) — they should land close; a divergence flags an engine-side config gap.
- **Grid:** `{vLLM, SGLang} × {base, MTP}` all at NVFP4 / 65536 / conc-32.
- **Repo choice — unsloth (no NVIDIA option).** Policy prefers an official `nvidia/` NVFP4 when one
  exists, but NVIDIA publishes none for the 27B (only the 35B-A3B sibling); 27B has only community NVFP4
  quants. Using **unsloth** (trusted quantizer per policy).
