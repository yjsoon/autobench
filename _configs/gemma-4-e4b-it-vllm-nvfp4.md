---
title: Gemma 4 E4B · vLLM · NVFP4
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
quant: NVFP4
quant_rationale: cosmicproc NVFP4 (W4A4) via NVIDIA Model Optimizer mixed-precision AutoQuantize — Per-Layer Embeddings + vision/audio towers kept BF16. First NVFP4 for the E-series (none from Google/unsloth); an individual quantizer (~61k downloads/mo, proper hf_quant_config.json) added at the user's explicit request. Comparison point against the done FP8 (leon-se) and BF16 base E4B runs.
source_repo: cosmicproc/gemma-4-E4B-it-NVFP4
download_url: https://huggingface.co/cosmicproc/gemma-4-E4B-it-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, NVFP4, ≤4B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). cosmicproc NVFP4 (W4A4, ModelOpt).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly cosmicproc/gemma-4-E4B-it-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model cosmicproc/gemma-4-E4B-it-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**NVFP4 (4-bit W4A4) point for the E4B on vLLM — the Blackwell-native 4-bit format.** cosmicproc's
NVFP4 quant (NVIDIA Model Optimizer, mixed-precision AutoQuantize; PLE + vision/audio towers left in
BF16), added at user request as the first NVFP4 for the elastic E-series. Direct comparison against the
done **FP8** (decode 869.7) and **BF16** (decode 565.8) base E4B runs on the same conc-32 ShareGPT
workload — does GB10's NVFP4 path beat FP8 on a ~4B model, or does the W4A4 activation quant cost more
than it saves at this size? **Pending run.** (Individual-uploader quant — flagged per the trusted-repo
policy, run on the user's explicit request.)
