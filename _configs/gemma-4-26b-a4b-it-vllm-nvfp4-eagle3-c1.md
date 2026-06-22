---
title: Gemma 4 26B-A4B · vLLM · NVFP4 + EAGLE3 · conc 1
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
concurrency: 1
tags: [gemma-4-26b-a4b, Google, Gemma, NVFP4, 16-40B, conc-1]
status: done
prefill_toks: 91.37
decode_toks: 48.0
mem_gb: 108.49
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-23
completed_at: 2026-06-23 00:38 +08
engine_image: vllm/vllm-openai:cu130-nightly@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
run_command: |
  # vllm/vllm-openai:cu130-nightly. NVFP4 base + RedHatAI EAGLE3 speculator (draft model).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-26B-A4B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"model":"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-26B-A4B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Conc-1 point for the Gemma 4 26B-A4B EAGLE3 sweep — moderate EAGLE3 acceptance (~2.0 accept-len),
healthy and clean (no harmony-style corruption).** NVIDIA NVFP4 base + RedHatAI EAGLE3 speculator on
vLLM (cu130-nightly), ctx 65536, conc 1.

- **Engine note — A/B settled on cu130-nightly.** The newer `nightly-aarch64` (vLLM 0.23.1) was tested
  and **cannot load Gemma-4 NVFP4** (`gemma4.py tie_weights → NotImplementedError` — the ModelOpt NVFP4
  quant method has no `tie_weights` for the lm_head/embed tie in 0.23.1). cu130-nightly (0.19.2) loads it
  fine, so it stays the engine for all Gemma vLLM runs. Image pinned in `engine_image`.
- **Load:** ready in **256 s** (NVFP4 weights + EAGLE3 head + CUDA-graph capture).
- **Workload:** ShareGPT V3, concurrency 1. **60/500 completed, 0 errors** before the **300 s time cap**.
- **Throughput:** decode **48.0 tok/s** (single stream), prefill **91.37 tok/s**. TTFT median **112 ms**,
  TPOT median **19.8 ms** — real latencies (no harmony buffering on this model).
- **EAGLE3 acceptance — moderate, ~2.0.** Across the run: **mean acceptance length ~1.8–2.3 (centered
  ~2.0)**, **avg draft acceptance ~25–43% (centered ~33%)**, per-position **~0.49–0.64 / 0.20–0.40 /
  0.07–0.25**. With `num_speculative_tokens=3` the head reliably lands the first draft token (~55–64%) but
  the 2nd/3rd decay fast. This is *well below* the same-size MTP runs (~3.0+) but *well above* the gpt-oss
  EAGLE3 collapse (~1.0–1.7) — a genuinely useful draft on this workload, and stable (no off-distribution
  corruption / 0 errors), unlike gpt-oss's harmony failures.
- **Memory: 108.5 GB is the vLLM `--gpu-memory-utilization 0.85` reservation** (NVFP4 weights ≈14 GB +
  EAGLE3 head), not the footprint.
- Compare decode + TPOT against the conc-32 run (decode 541 agg, the fastest Gemma config): at conc-1 the
  per-stream decode is 48, and acceptance should hold steady into c8/c32 (EAGLE3 acceptance is
  workload-driven).
