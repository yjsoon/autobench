---
title: gpt-oss-20b · vLLM · MXFP4
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: vLLM
quant: MXFP4
quant_rationale: gpt-oss's native FP4 format; FP4-accelerated with the CUTLASS kernels.
source_repo: openai/gpt-oss-20b
download_url: https://huggingface.co/openai/gpt-oss-20b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe]
status: done
prefill_toks: 654.13
decode_toks: 535.29
mem_gb: 107.83
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 06:00 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). Harmony vocab pre-seeded:
  # -v ~/models/tiktoken_cache:/vocab:ro  --env TIKTOKEN_ENCODINGS_BASE=/vocab  (see CLAUDE.md).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model openai/gpt-oss-20b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The vLLM half of the gpt-oss-20b engine comparison — much higher raw throughput than SGLang, but
with a harmony-parse robustness cost.** OpenAI's gpt-oss-20b (21B total / 3.6B active MoE, native
MXFP4), served on the cu130-nightly vLLM that NVIDIA documents for DGX Spark.

- **Workload:** ShareGPT V3, concurrency 32. **892/1000 completed, 108 errors**, in **418 s**
  (no time cap). Model loaded + CUDA-graph captured in **181 s**.
- **Aggregate throughput (conc 32):** prefill **654.1 tok/s**, decode **535.3 tok/s** — computed as
  total tokens ÷ wall-clock, so these are valid throughput figures over the 892 successful requests.
  **vs the SGLang run of the same model (359.7 / 279.5): vLLM is ~1.8× prefill and ~1.9× decode.**
  That's the headline of the engine comparison on GB10 — vLLM's fused MXFP4 MoE + CUDA-graph path
  pushes far more tokens/s here.
- **The caveats are real and specific to gpt-oss's harmony format on vLLM's chat path:**
  - **108 errors (~11%).** The streamed failures are harmony-parser faults —
    `Unexpected token 0 while expecting start token 200006 (<|start|>)` and
    `Unknown role: !comment…<|end|>`. With `--max-tokens 256`, gpt-oss is **truncated mid-reasoning**
    (completion ≈ 251 tok/req, right at the cap), leaving an incomplete harmony structure that vLLM's
    `stream_harmony` finalizer can't parse for ~1-in-9 requests. SGLang's `--reasoning-parser gpt-oss`
    tolerated the *same* 256-token truncation cleanly — a genuine engine robustness difference, not a
    config error.
  - **TTFT median 14.3 s / TPOT 0.0** are **not meaningful here.** vLLM's harmony chat path buffers the
    reasoning channel and emits the final message in one burst rather than streaming incremental
    content deltas, so our per-token client metrics collapse. (The aggregate tok/s above are unaffected
    — they don't depend on streaming granularity.)
- **Memory 107.8 GB is a reservation, not a footprint.** vLLM pre-allocates `--gpu-memory-utilization
  0.85` of the 121 GB unified pool as static KV; the actual MXFP4 weights are ~12 GB. Same caveat as
  every vLLM/SGLang config here.
- **Startup gotcha (now automated):** the harmony tiktoken vocab (`o200k_base.tiktoken`) is the run's
  *second* external fetch after the HF model pull, which the egress cap blocks → `HarmonyError` and no
  server. Fixed by pre-seeding `~/models/tiktoken_cache` and setting `TIKTOKEN_ENCODINGS_BASE=/vocab`
  (plain path, not `file://`). See CLAUDE.md.
