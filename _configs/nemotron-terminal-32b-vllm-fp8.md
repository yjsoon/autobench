---
title: Nemotron-Terminal-32B · vLLM · FP8
model: nvidia/Nemotron-Terminal-32B
company: NVIDIA
family: Nemotron
params: 33B (dense)
engine: vLLM
quant: FP8
quant_rationale: No official FP8 checkpoint is published — the repo ships BF16 (Qwen3 arch). This is vLLM's runtime/online FP8 (CutlassFP8ScaledMM) of the official BF16 weights; near-BF16 quality at half the bytes.
source_repo: nvidia/Nemotron-Terminal-32B
download_url: https://huggingface.co/nvidia/Nemotron-Terminal-32B
context: 40960
modalities: [text]
mm_served: true
tags: [NVIDIA, Nemotron, FP8, 16-40B]

status: done
prefill_toks: 170.66
decode_toks: 166.22
mem_gb: 106.64
mem_source: system MemAvailable delta (10s sampling) — vLLM 0.85 reservation; real FP8 weights 32 GB, see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 01:28 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly. BF16 weights, FP8-quantized at load by vLLM
  # (--quantization fp8). NOTE ctx=40960 — the model's native max_position_embeddings;
  # requesting 65536 fails ModelConfig validation (would need VLLM_ALLOW_LONG_MAX_MODEL_LEN).
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly \
    nvidia/Nemotron-Terminal-32B \
    --host 0.0.0.0 --port 8000 --max-model-len 40960 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --trust-remote-code --quantization fp8
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Nemotron-Terminal-32B \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**A dense 32B on runtime FP8 — and the engine-comparison data point against the FP4 MoEs.**
NVIDIA's agentic/terminal-tuned model (Qwen3 architecture, `Qwen3ForCausalLM`). **No official FP8
checkpoint exists** — the repo is BF16 — so this is vLLM's **online FP8** quantization
(`Fp8OnlineLinearMethod`, `CutlassFP8ScaledMMLinearKernel`) of the official BF16 weights, a standard
and trustworthy path. Served on `vllm/vllm-openai:cu130-nightly`.

- **Context gotcha:** the model's native `max_position_embeddings` is **40960**, so this run uses
  ctx 40960 (not the 65536 the other configs use). Requesting 65536 hard-fails vLLM's ModelConfig
  validation (RoPE would produce NaNs past the trained length) unless you force
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` — not done here, since extrapolating context isn't the point.
- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min time cap** at **626/1000 prompts,
  0 errors** (938 s).
- **Throughput (aggregate, conc 32):** prefill **170.7 tok/s**, decode **166.2 tok/s**. This sits
  exactly where a dense 32B should: faster than the 120B giants (gpt-oss-120b 140, Super-120B 97) but
  far slower than the 3B-active 30B MoEs (Elastic 353, Nano-Omni 389) — because *every* one of its
  32B params is active per token, vs 3B for the MoEs. TPOT median **170.8 ms** (≈5.9 tok/s/stream).
- **TTFT is dramatically lower than the NVFP4 MoE runs — 605 ms** (vs 13–78 s). Two reasons: vLLM
  enabled **prefix caching** for this model (`enable_prefix_caching=True`, off for the NVFP4 configs),
  and with only 6.6× KV concurrency headroom the scheduler admits prompts more gradually rather than
  dumping 32 big prefills at once. So per-request latency here is genuinely good even though aggregate
  decode is mid-pack — a nice illustration that TTFT is a scheduling property, not just a model-size one.
- **Memory:** real FP8 footprint **32.0 GiB** weights (33B × ~1 byte). KV pool 66.4 GiB but
  `kv_cache_dtype=auto` (bf16, *not* fp8 like the NVFP4 models) → only **272,048 tokens** = 6.6×
  concurrency at 40960 ctx (dense attention + bf16 KV is KV-heavy). The 106.6 GB MemAvailable delta is
  the `--gpu-memory-utilization 0.85` reservation.
- **Startup (cold): ready after 1004 s** — 494 s download + 412 s weight load (the BF16 re-downloaded
  after the first ctx-65536 launch failed validation) + 76 s engine init (27 s compilation, incl. the
  online FP8 conversion). Subsequent launches skip the download.
