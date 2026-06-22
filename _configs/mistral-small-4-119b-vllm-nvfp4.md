---
title: Mistral Small 4 119B · vLLM · NVFP4
model: mistralai/Mistral-Small-4-119B-2603-NVFP4
company: Mistral AI
family: Mistral
params: 119B (dense, MLA)
engine: vLLM
quant: NVFP4
quant_rationale: Blackwell-native FP4, hardware-accelerated on GB10. Mistral's OWN official NVFP4 build of Small-4-119B (the base repo is FP8; this is the separate -NVFP4 checkpoint) — trusted source.
source_repo: mistralai/Mistral-Small-4-119B-2603-NVFP4
download_url: https://huggingface.co/mistralai/Mistral-Small-4-119B-2603-NVFP4
context: 65536
modalities: [text, image]
mm_served: true
tags: [mistral-small-4-119b, Mistral AI, Mistral, NVFP4, 41-130B]
status: done
prefill_toks: 174.3
decode_toks: 134.96
mem_gb: 116.2
mem_source: system MemAvailable delta (10s sampling) — vLLM 0.90 reservation; real weights 66.22 GB, see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 02:50 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly. Mistral NATIVE format (params.json/consolidated/tekken)
  # → needs the mistral tokenizer/config/load flags. The model is MLA (DeepSeek-V2-style) and
  # vLLM's Triton MLA decode kernel FAILS TO COMPILE on GB10 — must disable MLA + raise util.
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    --env VLLM_MLA_DISABLE=1 \
    vllm/vllm-openai:cu130-nightly \
    mistralai/Mistral-Small-4-119B-2603-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.90 --max-num-seqs 32 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model mistralai/Mistral-Small-4-119B-2603-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The last headliner — and the one that fought back.** Mistral's own NVFP4 build of Small-4-119B.
Two non-obvious things had to be solved to get a number, both recorded in
[CLAUDE.md](https://github.com/gauravmm/autobench) for the next run:

1. **Native Mistral format.** The repo ships `params.json` / `consolidated-*.safetensors` /
   `tekken.json` (no `config.json`), so vLLM needs `--tokenizer-mode mistral --config-format mistral
   --load-format mistral`. vLLM detects the quant as **compressed-tensors NVFP4** and serves it with
   `FlashInferCutlassNvFp4LinearKernel`.
2. **The Triton MLA decode kernel crashes on GB10.** This model uses **MLA** (Multi-head Latent
   Attention, DeepSeek-V2-style) with a Pixtral vision tower. vLLM's default Triton MLA decode kernel
   **fails to compile** here — *every* request killed EngineCore with
   `triton_decode_attention … Cannot make_shape_compatible: incompatible dimensions 256 and 512`
   (the first full run scored **1000/1000 errors**). `VLLM_ATTENTION_BACKEND=FLASHINFER_MLA` is
   silently ignored (falls back to Triton). The fix that works: **`VLLM_MLA_DISABLE=1`** — use
   standard attention and bypass the broken kernel. Because MLA-disabled materializes the *full* KV
   (vs MLA's compressed latent), 0.85 util under-sizes the pool for 65536 ctx (needs 36 GiB, had
   33.8), so also bump **`--gpu-memory-utilization 0.90`**.

With those two fixes it serves cleanly:

- **Workload:** ShareGPT V3, concurrency 32. **Hit the 15-min cap** at **574/1000 prompts, 0 errors**
  (947 s).
- **Throughput (aggregate, conc 32):** prefill **174.3 tok/s**, decode **135.0 tok/s** — right in the
  120B-giant band (gpt-oss-120b 140, this 135, Super-120B 97). TTFT median **743 ms** (low, like the
  other prefix-caching vLLM runs — not the 13–78 s of the NVFP4 Nemotron MoEs), TPOT median **219 ms**
  (≈4.6 tok/s/stream — slow, as expected for a ~120B with every param active per token).
- **Memory — MLA-disabled is KV-expensive.** Real NVFP4 weights **66.22 GiB** (logged). With MLA
  disabled the full-attention KV pool gives only **1.13× concurrency at the full 65536 ctx** (it can
  barely hold one max-length sequence) — fine here only because ShareGPT prompts are short and paged
  KV packs many of them. The 116.2 GB MemAvailable delta is the `--gpu-memory-utilization 0.90`
  reservation (idle avail dropped to ~5.5 GB — this run fills the box). **Caveat:** the MLA-disable
  workaround costs both KV capacity and decode efficiency; a future vLLM that compiles the MLA kernel
  on GB10 would likely serve this model faster and with far more context headroom. Worth re-running
  then.
- Startup (cold, first download): ready after **1248 s** (602 s download + 533 s load of 66 GiB +
  init). `quantization=compressed-tensors`, `enable_prefix_caching=True`.
