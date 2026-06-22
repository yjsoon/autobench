---
title: Nemotron-3 Super 120B · vLLM · NVFP4
model: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
company: NVIDIA
family: Nemotron
params: 123B / 12B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4 via FlashInfer/CUTLASS — NVIDIA's own quant; the 120B-class NVFP4 flagship. (config.json declares quant_algo MIXED_PRECISION → see Notes for the container caveat.)
source_repo: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
download_url: https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [nvidia-nemotron-3-super-120b-a12b, NVIDIA, Nemotron, NVFP4, 41-130B, Spark recipe, conc-32]
status: done
prefill_toks: 115.17
decode_toks: 96.91
mem_gb: 107.63
mem_source: system MemAvailable delta (10s sampling) — vLLM 0.85 reservation; real weights 69.54 GB, see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 00:22 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"] → pass <model> <flags>).
  # The cu130-nightly serves this NVFP4 checkpoint fine; an OLDER NGC vLLM container
  # rejected its quant_algo=MIXED_PRECISION against a hardcoded whitelist — use this image.
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly \
    nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --trust-remote-code --reasoning-parser nemotron_v3
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The 120B-class NVFP4 flagship.** Served on the documented Spark stack
(`vllm/vllm-openai:cu130-nightly`, [vLLM DGX Spark
blog](https://vllm-project.github.io/2026/06/01/vllm-dgx-spark.html)). The image handles the NVFP4
checkpoint cleanly — note that an **older NGC vLLM container rejected this model's
`quant_algo: MIXED_PRECISION`** against a hardcoded whitelist
([vllm#37854](https://github.com/vllm-project/vllm/issues/37854)); cu130-nightly does not.

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min time cap** at **361/1000 prompts,
  0 errors** (938 s). The heaviest model benchmarked so far — like gpt-oss-120b, 1000 prompts don't
  fit the cap.
- **Throughput (aggregate, conc 32):** prefill **115.2 tok/s**, decode **96.9 tok/s**. This is the
  whole-system aggregate at 32-way load; the [vLLM blog](https://vllm-project.github.io/2026/06/01/vllm-dgx-spark.html)
  reports ~22–24 tok/s *single-stream* decode for the same model — consistent (aggregate over a
  batch is higher than one stream). Still the slowest config to date, as expected for 12B active
  params (vs gpt-oss-120b's 5.1B → 140 tok/s; this one's 12B → 97 tok/s tracks the active-param ratio).
- **Severe queueing at conc 32 — two metric artifacts worth reading carefully:**
  - Median **TTFT 78.7 s**: with `--max-num-seqs 32`, 32 long ShareGPT prompts are admitted at once
    and their prefills queue behind each other on a 12B-active model with limited prefill bandwidth,
    so each stream waits ~78 s before its first token. (NVIDIA's own recipe uses `--max-num-seqs 4`
    for this model precisely to bound this.)
  - **TPOT reads 0.0 ms** — an artifact, not real instant decode: under this overload vLLM delivers a
    queued stream's tokens in a burst right before it finishes, so the first streamed token lands
    ≈ the last, collapsing the measured inter-token gap. The honest decode-rate signal here is the
    **aggregate 96.9 tok/s**, not TPOT.
  - Throughput-wise the run is valid; latency is the casualty of the fixed conc-32 methodology on a
    model this size. A `--max-num-seqs 4` rerun would give blog-comparable per-stream latency at
    lower aggregate — out of scope for the speed-only methodology, but flagged.
- **Memory — nearly fills the box.** Real footprint from logs: **NVFP4 weights 69.54 GiB**, leaving
  only **29.98 GiB** for the (fp8_e4m3) KV pool = **1,306,240 tokens** (72× concurrency at 64K — still
  plenty for short ShareGPT turns, but the tightest KV headroom of the NVFP4 set). The 107.6 GB
  MemAvailable delta is the `--gpu-memory-utilization 0.85` reservation.
- **CUDA-graph capture: 4 s, 0.13 GiB.** **Startup (cold, first download): ready after 1285 s** —
  667 s weight download + 516 s load (1192 s / 69.5 GiB total) + 42 s engine init (17.6 s compilation).
  The ~62 GB download dominates; subsequent launches skip it. `quantization=modelopt_mixed`.
