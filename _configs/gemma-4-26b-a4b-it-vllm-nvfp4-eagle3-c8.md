---
title: Gemma 4 26B-A4B · vLLM · NVFP4 + EAGLE3 · conc 8
model: google/gemma-4-26B-A4B-it
company: Google
family: Gemma
params: 26B / 4B (MoE)
engine: vLLM
speculative: EAGLE3
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + RedHatAI's official EAGLE3 speculator — the fast quant plus lossless speculative decoding stacked together.
source_repo: nvidia/Gemma-4-26B-A4B-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 8
tags: [gemma-4-26b-a4b, Google, Gemma, NVFP4, 16-40B, conc-8]
status: done
prefill_toks: 307.78
decode_toks: 211.48
mem_gb: 108.23
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-23
completed_at: 2026-06-23 00:48 +08
engine_image: vllm/vllm-openai:cu130-nightly@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
run_command: |
  # vllm/vllm-openai:cu130-nightly. NVFP4 base + RedHatAI EAGLE3 speculator (draft model).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-26B-A4B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"model":"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-26B-A4B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Conc-8 point for the Gemma 4 26B-A4B EAGLE3 sweep — acceptance steady at ~2.15, confirming it's
workload-driven; 0 errors.** NVIDIA NVFP4 base + RedHatAI EAGLE3 speculator on vLLM (cu130-nightly),
ctx 65536, conc 8.

- **Load:** ready in **290 s**.
- **Workload:** ShareGPT V3, concurrency 8. **278/500 completed, 0 errors** before the **300 s time cap**.
- **Throughput:** prefill **307.78 tok/s**, decode **211.48 tok/s** aggregate (~26.4 tok/s/stream). TTFT
  median **237 ms**, TPOT median **35.0 ms**.
- **EAGLE3 acceptance — steady ~2.15.** Run-aggregate **mean acceptance length ~2.0–2.33 (centered ~2.15)**,
  **avg draft acceptance ~33–44% (centered ~38%)**, per-position **~0.54–0.66 / 0.29–0.42 / 0.15–0.25** —
  essentially unchanged from conc-1 (~2.0), confirming EAGLE3 acceptance is workload- not
  concurrency-driven here (the correct behavior, unlike gpt-oss). Drafts ~330 tok/s, accepts ~120 tok/s.
- **Memory: 108.2 GB** = vLLM 0.85 reservation + EAGLE3 head, not footprint.
- **Sweep shape:** decode 48 (c1) → 211 agg (c8) → 541 agg (c32) — aggregate throughput scales with batch
  while per-stream decode falls (48 → 26.4 → ~17), the expected trade-off. Acceptance ~2.0–2.15 throughout.
  The EAGLE3 head is a real ~2× draft on ShareGPT (vs MTP's ~3× on the dense Gemmas), and rock-stable.
