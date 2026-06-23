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
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:
measured_on: 2026-06-23
completed_at:
engine_image: lmsysorg/sglang:spark
run_command: |
  # INTENDED (not yet run). SGLang NEXTN/MTP. Same ModelOpt-NVFP4-on-SGLang risk as the base sibling —
  # verify the format loads before trusting any spec result; BLOCK if SGLang rejects it.
  docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path nvidia/Qwen3.6-35B-A3B-NVFP4 --host 0.0.0.0 --port 30000 \
    --context-length 65536 --trust-remote-code \
    --speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 4
  python3 scripts/bench-serving.py --base-url http://localhost:30000 --model nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**BLOCKED 2026-06-23 — same SGLang `spark` arch wall.** transformers 4.57.1 in `lmsysorg/sglang:spark`
can't load the Qwen3.6 (`qwen3_5`) arch (measured on `qwen3-6-27b-nvfp4-sglang`); MTP is moot until the
base loads. Revisit with a newer SGLang tag. The MoE NVFP4 grid is therefore vLLM-only for now.

---

**Was queued — Qwen3.6-35B-A3B NVFP4 + MTP on SGLang (NEXTN), NVIDIA official quant.** Completes the MoE
NVFP4 grid: `{vLLM, SGLang} × {base, MTP}`.

- **Repo — NVIDIA official:** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4).
- **MTP via NEXTN:** `--speculative-algo NEXTN --speculative-num-steps 3 --speculative-eagle-topk 1
  --speculative-num-draft-tokens 4`. **Same ModelOpt-NVFP4-on-SGLang risk** as the base sibling — verify
  the format loads first; BLOCK if SGLang rejects it.
- **Acceptance:** capture SGLang accept length, cross-check vs the vLLM MTP run (same draft, different engine).
- **conc-8/conc-1 variants** to be added when this spec config is benchmarked.
