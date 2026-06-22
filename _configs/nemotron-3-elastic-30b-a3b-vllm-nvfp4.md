---
title: Nemotron-3 Elastic 30B-A3B · vLLM · NVFP4
model: nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4
company: NVIDIA
family: Nemotron
params: 30B / 3B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4, hardware-accelerated on GB10 via FlashInfer/CUTLASS NVFP4 kernels — NVIDIA's own quant for its own model; first choice for the Nemotron line.
source_repo: nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [nvidia-nemotron-labs-3-elastic-30b-a3b, NVIDIA, Nemotron, NVFP4, 16-40B, Spark recipe, conc-32]
status: done
prefill_toks: 391.26
decode_toks: 353.0
mem_gb: 111.16
mem_source: system MemAvailable delta (10s sampling) — vLLM 0.85 KV reservation; real weights 18.65 GB, see Notes
measured_on: 2026-06-21
completed_at: 2026-06-21 23:23 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly — the image NVIDIA/vLLM document for DGX Spark.
  # ENTRYPOINT is ["vllm","serve"], so the command is just <model> <flags>.
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly \
    nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --trust-remote-code --reasoning-parser nemotron_v3
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/NVIDIA-Nemotron-Labs-3-Elastic-30B-A3B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**First NVFP4 headliner — the Blackwell-native FP4 path, validated.** vLLM picks up the NVFP4
weights with `FlashInferCutlassNvFp4LinearKernel` (GEMM) and the `FLASHINFER_CUTLASS` MoE backend,
auto-tuning the fp4_gemm kernels at startup. This is the documented Spark serving stack for the
Nemotron NVFP4 models (`vllm/vllm-openai:cu130-nightly`, [vLLM DGX Spark
blog](https://vllm-project.github.io/2026/06/01/vllm-dgx-spark.html)).

- **Workload:** ShareGPT V3 at concurrency 32. **Completed 1000/1000 with 0 errors** in **696 s**
  (no time-cap) — comfortably inside the 15-min budget, unlike the 120b gpt-oss.
- **Throughput (aggregate, conc 32):** prefill **391.3 tok/s**, decode **353.0 tok/s** — the best
  decode of any config so far (vs gpt-oss-20b 279.5, gpt-oss-120b 140.3), as expected for a 3B-active
  MoE on the FP4-accelerated path.
- **High TTFT is a batching artifact, not a stall.** Median TTFT **13.2 s** (vs ~0.3 s for gpt-oss-20b
  on SGLang) while TPOT is an excellent **26.6 ms** (≈38 tok/s/stream). With `--max-num-seqs 32` and
  32 concurrent long ShareGPT prompts admitted at once, each request's *prefill* queues behind the
  others before its first token — so per-request TTFT inflates even though aggregate prefill/decode
  throughput stays high. Our scope is throughput, so this is expected; flagged because it's a big
  TTFT swing vs the SGLang runs (which schedule prefills more incrementally).
- **Memory — vLLM reports the real footprint (unlike SGLang).** The 111 GB MemAvailable delta is the
  `--gpu-memory-utilization 0.85` reservation, but vLLM logs the actual breakdown: **model weights
  18.65 GiB** (the true NVFP4 footprint for 30B), then **80.9 GiB** left for the KV pool =
  **2,827,504 tokens** (≈187× concurrency headroom at 64K ctx — this model has enormous room to grow
  context or batch). CUDA-graph pool just 0.26 GiB.
- **Startup (cold, first download): ready after 411 s** — 166 s weight download + 111 s weight load
  (281 s / 18.65 GiB total for model loading) + **CUDA-graph capture 11 s** (18 graphs: 11 PIECEWISE
  + 7 FULL) + 74 s engine init (11 s compilation). The graph capture is trivial here vs the gpt-oss-120b's
  32.7 s. Subsequent launches skip the download.
- Flags: `--trust-remote-code --reasoning-parser nemotron_v3` (Nemotron-3 reasoning format).
