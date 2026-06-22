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
status: done
prefill_toks: 1284.1
decode_toks: 1073.8
mem_gb: 110.24
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 21:57 +08
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

**The new fastest decode in the entire benchmark — NVFP4 beats FP8 on the tiny E4B.** cosmicproc's
NVFP4 quant (NVIDIA Model Optimizer, mixed-precision AutoQuantize; PLE + vision/audio towers left in
BF16) on vLLM, added at user request as the first NVFP4 for the elastic E-series.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **206.8 s** — the quickest full
  run of any config (edges out FP8's 253 s). TTFT median **82.7 ms**, TPOT median **27.3 ms**
  (~37 tok/s/stream), req throughput **4.84/s**.
- **Throughput (aggregate, conc 32):** prefill **1284.1 tok/s**, decode **1073.8 tok/s** — **the highest
  decode measured anywhere in this benchmark**, ahead of the FP8 run (the prior record holder).

  | Engine / quant | prefill | decode |
  |---|---|---|
  | llama.cpp Q4_K_M | 329.0 | 435.0 |
  | vLLM BF16 | 678.8 | 565.8 |
  | vLLM FP8 | 1047.2 | 869.7 |
  | **vLLM NVFP4** | **1284.1** | **1073.8** |

  On GB10's Blackwell tensor cores the **NVFP4 (W4A4) path is a clear win even at this tiny size** —
  ~23% faster decode and ~23% faster prefill than FP8, despite the activation quant. vLLM uses
  `FlashInferCutlassNvFp4LinearKernel` for the NVFP4 GEMMs; attention falls back to `TRITON_ATTN`
  (forced because Gemma4 has heterogeneous head dims — `head_dim=256` local / `512` global).
- **Memory: 110.2 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (NVFP4 weights ≈ 2–3 GB; the reservation dwarfs the model at this size — same as the FP8/BF16 runs).
- **Load gotcha:** the **first launch hung at EngineCore init** (0.17% CPU, weights never loaded — a
  transient spawn deadlock on this vLLM 0.19.2rc1 build; **no** NVFP4 incompatibility). A clean retry
  loaded fine (**ready after 363 s**, weight load + Triton/CUDA-graph capture). If a future run hangs at
  `Enabled custom fusions: act_quant` with no EngineCore log, just kill and relaunch.
- Text path benchmarked (`mm_served: false`). Individual-uploader quant — flagged per the trusted-repo
  policy, run on the user's explicit request; the strong, error-free result corroborates the checkpoint.
