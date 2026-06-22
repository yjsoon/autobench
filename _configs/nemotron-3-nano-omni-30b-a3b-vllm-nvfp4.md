---
title: Nemotron-3 Nano-Omni 30B-A3B · vLLM · NVFP4
model: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
company: NVIDIA
family: Nemotron
params: 33B / 3B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4 via FlashInfer/CUTLASS — NVIDIA's own quant; this is the multimodal (audio-video-language) Nano in its NVFP4 build.
source_repo: nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
download_url: https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4
context: 65536
modalities: [text, image, audio, video]
mm_served: true
concurrency: 32
tags: [nemotron-3-nano-omni-30b-a3b-reasoning, NVIDIA, Nemotron, NVFP4, 16-40B, Spark recipe, conc-32]
status: done
prefill_toks: 444.92
decode_toks: 388.93
mem_gb: 112.56
mem_source: system MemAvailable delta (10s sampling) — vLLM 0.85 KV reservation; real weights 21.5 GB, see Notes
measured_on: 2026-06-21
completed_at: 2026-06-21 23:43 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"] → pass <model> <flags>)
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly \
    nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --trust-remote-code --reasoning-parser nemotron_v3
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The "best Spark showcase" NVFP4 Nano — and the fastest decode in the set so far.** This is the
full **AVLM** (audio-video-language): vLLM loads a RADIO vision encoder *and* a sound encoder
("Found sound config, initializing sound encoder for Nemotron AVLM") alongside the 3B-active MoE,
even though we benchmark text-only ShareGPT. Served on the documented Spark stack
(`vllm/vllm-openai:cu130-nightly`).

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **612 s** (no time-cap).
- **Throughput (aggregate, conc 32):** prefill **444.9 tok/s**, decode **388.9 tok/s** — the **best
  decode of any config to date** (Elastic-30B 353.0 → Nano-Omni 388.9), and TPOT median **11.1 ms**
  (≈90 tok/s/stream) is excellent. The fp8 KV cache (below) is a big part of why it edges out the
  Elastic.
- **Same high-TTFT batching artifact:** median TTFT **16.0 s** (32 long prompts admitted at once,
  prefills queue) with a tiny 11 ms TPOT — throughput is unaffected; per-request latency is the
  trade. Consistent with the Elastic-30B (13.2 s); inherent to vLLM admitting `--max-num-seqs 32`
  big prefills together.
- **Memory — fp8 KV cache doubles the token budget.** Real footprint from the logs: **weights +
  encoders 21.5 GiB** (vs Elastic's 18.65 — the extra is the vision/sound encoders), and vLLM runs
  this model with **`kv_cache_dtype=fp8_e4m3`**, so the 74.78 GiB KV pool holds **5,226,368 tokens**
  = **350× concurrency** at 64K ctx (vs Elastic's 2.83M / 187×). The 112.6 GB MemAvailable delta is
  again the `--gpu-memory-utilization 0.85` reservation, not the footprint.
- **CUDA-graph capture: 2 s, 0.11 GiB** — the fastest capture in the set (capture sizes 1…64).
- **Startup (cold, first download): ready after 397 s** — 246 s weight download + 45 s load (296 s /
  21.5 GiB model loading total) + 43 s engine init (7.8 s compilation).
- ModelOpt flags the checkpoint as an **experimental NVFP4 format** ("could change in future") —
  worth re-verifying on a later vLLM. `quantization=modelopt_mixed`. Flags: `--trust-remote-code
  --reasoning-parser nemotron_v3`.
