---
title: gpt-oss-120b · SGLang · MXFP4
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: SGLang
quant: MXFP4
quant_rationale: gpt-oss's native MXFP4 (~66 GB resident), served from the original safetensors by SGLang — the documented SOTA engine for gpt-oss on DGX Spark.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe]
status: done
prefill_toks: 187.68
decode_toks: 140.27
mem_gb: 112.46
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation, see Notes
measured_on: 2026-06-21
completed_at: 2026-06-21 22:56 +08
run_command: |
  # lmsysorg/sglang:spark (the documented SOTA engine for gpt-oss on DGX Spark)
  docker run --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/tiktoken_encodings:/tiktoken_encodings \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b --host 0.0.0.0 --port 30000 \
    --context-length 65536 --reasoning-parser gpt-oss --tool-call-parser gpt-oss
  python3 scripts/bench-serving.py --base-url http://localhost:30000 \
    --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The flagship Spark-class headliner, on its recommended engine.** Same path as the 20b
(llama.cpp can't serve gpt-oss's harmony chat format — see the blocked llama.cpp configs),
so SGLang is the real number ([LMSYS](https://www.lmsys.org/blog/2025-11-03-gpt-oss-on-nvidia-dgx-spark/)).

- **Workload:** ShareGPT V3 at concurrency 32. **Hit the 15-min time cap** at **560/1000 prompts,
  0 errors** (970 s wall — the last in-flight request finished just past the 900 s deadline). The
  120b is heavy enough that 1000 prompts don't fit the cap; 560 completions is still a stable
  aggregate. The `hit_time_cap` flag is set — this is a slow path worth flagging vs. the 20b
  (which finished all 1000 in 859 s).
- **Throughput (aggregate, conc 32):** prefill **187.7 tok/s**, decode **140.3 tok/s** — roughly
  half the 20b's (359.7 / 279.5), tracking the ~1.4× more active params (5.1B vs 3.6B) plus the
  larger weight/KV footprint. Per-stream: TTFT median **918 ms**, TPOT median **157 ms**
  (≈6.4 tok/s/stream under 32-way load).
- **Startup cost is dominated by weight load, not graph capture.** Cold launch to "ready to roll"
  was **443 s**, broken down from the container logs:
  - **Weight load: ~375 s** — 66.4 GB of MXFP4 weights streamed from the (already-cached) HF repo
    into unified memory at ~180 MB/s. This is the dominant cost and it's **disk-read-bound**, not
    compute — even with the model fully cached locally. (First-ever launch additionally downloads
    the ~120 GB repo.)
  - **KV cache alloc:** 513,893 tokens, K 17.64 GB + V 17.64 GB = **35.3 GB** static KV pool.
  - **CUDA-graph capture: 32.7 s** — SGLang captured **36 batch sizes** (bs = 1 … 256) for the MoE
    decode path. Notable, but small next to the 6-minute weight load. (`max_total_num_tokens=513893`,
    `max_running_requests=4014`, post-capture available memory **8.05 GB**.)
- **Memory caveat — 112 GB is a reservation, not the footprint.** As with the 20b, SGLang
  pre-allocates a static fraction of unified memory (`--mem-fraction-static`, default ≈0.9), so the
  MemAvailable delta captures the reservation. The *actual* resident breakdown here is concrete from
  the logs: **66.4 GB weights + 35.3 GB KV + 3.4 GB CUDA graphs ≈ 105 GB**, leaving ~8 GB headroom
  of the 121 GB. So unlike the 20b (where 112 GB hugely overstated ~12 GB of weights), for the 120b
  the reservation is genuinely close to the real footprint — it nearly fills the box. Little room to
  raise context or `--mem-fraction-static` further.
- Requires the tiktoken encodings (`o200k_base`, `cl100k_base`) mounted at `TIKTOKEN_ENCODINGS_BASE`,
  plus `--reasoning-parser gpt-oss --tool-call-parser gpt-oss`.
