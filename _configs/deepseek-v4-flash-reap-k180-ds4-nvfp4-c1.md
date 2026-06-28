---
title: DeepSeek V4-Flash REAP-K180 · ds4 · NVFP4-hybrid · conc 1 (1M ctx)
model: sleepyeldrazi/DeepSeek-v4-Flash-REAP-K180-NVFP4
company: DeepSeek
family: DeepSeek
params: DeepSeek V4-Flash, REAP-K180 expert-pruned MoE (105.9 GB hybrid-quant GGUF)
engine: ds4 (ds4-nvfp4-spark, custom GB10 runtime)
speculative: none — ds4 is a serial single-worker; the DeepSeek MTP head is not exercised by this runtime
quant: NVFP4-hybrid (NVFP4 experts + Q2_K + Q8_0 + F16)
quant_rationale: The only published artifact for this model is the author's own hybrid-quant GGUF (arch `deepseek4`), paired with their bespoke `ds4-nvfp4-spark` runtime — no vLLM/llama.cpp path exists. This is the FIRST config of this model to actually run on the Spark (the unpruned `deepseek-v4-flash-vllm-*` configs are all blocked on fit/kernel walls — see those pages).
source_repo: sleepyeldrazi/DeepSeek-v4-Flash-REAP-K180-NVFP4
download_url: https://huggingface.co/sleepyeldrazi/DeepSeek-v4-Flash-REAP-K180-NVFP4
context: 1048576
modalities: [text]
mm_served: false
concurrency: 1
tags: [deepseek-v4-flash-reap-k180, DeepSeek, NVFP4, 130B+, conc-1, Spark recipe]
status: done
prefill_toks: 18.52
decode_toks: 11.39
mem_gb: 107.8
mem_source: ds4 managed-model resident (98.64 GiB, engine log) + context buffers 9.14 GiB at ctx=1048576 (FP8 KV-turbo, compressed_kv_rows=262146); system `free -h` showed 102 GiB used incl ~3.4 GiB OS baseline
measured_on: 2026-06-28
completed_at: 2026-06-28 13:34 +08
engine_image: ds4-flash:local — built from github.com/sleepyeldrazi/ds4-nvfp4-spark @ 2060b3def7d7555b3e0a6c5b0cc374ba49702c96, base nvidia/cuda:13.0.1-devel-ubuntu24.04, `make CUDA_ARCH=sm_121a`
run_command: |
  # Custom runtime — the author publishes NO image, so we build one (single-stage, CUDA 13.0.1 devel
  # already ships nvcc + SBSA/arm64 cudart+cublas; 13.0.1 supports compute_121 — no need for 13.2).
  # Dockerfile (full copy in this repo's notes / the gateway's ~/ds4flash/Dockerfile):
  #   FROM nvidia/cuda:13.0.1-devel-ubuntu24.04
  #   ARG DS4_COMMIT=2060b3def7d7555b3e0a6c5b0cc374ba49702c96
  #   RUN apt-get update && apt-get install -y git ca-certificates build-essential
  #   RUN git clone https://github.com/sleepyeldrazi/ds4-nvfp4-spark.git && cd ds4-nvfp4-spark \
  #       && git checkout ${DS4_COMMIT} && make CUDA_ARCH=sm_121a -j"$(nproc)" all \
  #       && cp ds4 ds4-server ds4-bench /usr/local/bin/
  #   ENTRYPOINT ["ds4-server"]
  #
  # The model needs ~99 GiB resident — it CANNOT coexist with any other engine on the 121 GB box.
  docker run -d --name serving-ds4-flash --gpus all \
    -e DS4_CUDA_MANAGED_MODEL=1 -e DS4_KV_TURBO=1 \
    -v ~/ds4flash/gguf:/models:ro -p 127.0.0.1:8000:8000 \
    ds4-flash:local \
    -m /models/DeepSeek-V4-Flash-REAP-K180-hybrid.gguf --host 0.0.0.0 --port 8000 --ctx 1048576
  python3 scripts/bench-serving.py --base-url http://127.0.0.1:8000 --model deepseek-v4-flash \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 1 --max-tokens 256
---

**The honest single-stream point for this model — ~12.5 tok/s decode, and the full 1M context fits with
room to spare.** Author's REAP-K180 hybrid GGUF on the bespoke `ds4-nvfp4-spark` runtime (GB10, sm_121a),
ctx **1,048,576**, conc 1. This and its conc-32 sibling are the **first completed numbers for any
DeepSeek-V4-Flash config on the Spark** — every `vllm-*` attempt was blocked on a fit/kernel wall.

- **Load:** ready in **135 s** via CUDA managed memory (`DS4_CUDA_MANAGED_MODEL=1`); the 105.9 GB GGUF maps
  to **98.64 GiB resident** on device.
- **1M context is NOT memory-constrained here.** With `DS4_KV_TURBO=1` (FP8 KV) the runtime reports
  `context buffers 9136.84 MiB (ctx=1048576 … compressed_kv_rows=262146)` — i.e. ~4:1 KV compression, so the
  entire 1M window costs only **~9.1 GiB**. Weights + KV ≈ **107.8 GiB**, leaving ~19 GiB free. No need to
  trim the context to fit.
- **Workload:** ShareGPT V3, conc 1. **42/1000 completed, 0 errors** before the **900 s time cap**
  (`hit_time_cap=true`, ran 907 s).
- **Throughput:** decode **11.39 tok/s** (single stream), prefill **18.52 tok/s**. TTFT median **730 ms**,
  TPOT median **79.6 ms** (≈ **12.6 tok/s/req** in steady state — matches the author's ~12 tok/s claim and
  the live engine log's ~12.5 t/s decode chunks).
- **No speculative path.** The runtime decodes one token at a time on a single worker; the DeepSeek MTP head
  is not used. There is no draft/EAGLE option in `ds4-server`, so spec-decode is N/A for this engine (unlike
  the gateway's main Qwen engine, which runs MTP).
- **This is the configuration left running on the gateway box** (`serving-ds4-flash`, `127.0.0.1:8000`,
  served name `deepseek-v4-flash`) after the sweep. The standard vLLM stack is down while it's up — they
  can't share 121 GB.

See the **conc-32 sibling** for why concurrency buys nothing here: `ds4-server` is serial, so 32 in-flight
requests just build a ~9.6-minute queue at the same ~11 tok/s aggregate.
