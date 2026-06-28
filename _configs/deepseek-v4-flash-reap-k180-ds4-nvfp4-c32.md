---
title: DeepSeek V4-Flash REAP-K180 · ds4 · NVFP4-hybrid · conc 32 (32K ctx)
model: sleepyeldrazi/DeepSeek-v4-Flash-REAP-K180-NVFP4
company: DeepSeek
family: DeepSeek
params: DeepSeek V4-Flash, REAP-K180 expert-pruned MoE (105.9 GB hybrid-quant GGUF)
engine: ds4 (ds4-nvfp4-spark, custom GB10 runtime)
speculative: none — ds4 is a serial single-worker; the DeepSeek MTP head is not exercised by this runtime
quant: NVFP4-hybrid (NVFP4 experts + Q2_K + Q8_0 + F16)
quant_rationale: The only published artifact for this model is the author's own hybrid-quant GGUF (arch `deepseek4`), paired with their bespoke `ds4-nvfp4-spark` runtime — no vLLM/llama.cpp path exists. First config of this model to actually run on the Spark (the unpruned `deepseek-v4-flash-vllm-*` configs are all blocked — see those pages).
source_repo: sleepyeldrazi/DeepSeek-v4-Flash-REAP-K180-NVFP4
download_url: https://huggingface.co/sleepyeldrazi/DeepSeek-v4-Flash-REAP-K180-NVFP4
context: 32768
modalities: [text]
mm_served: false
concurrency: 32
tags: [deepseek-v4-flash-reap-k180, DeepSeek, NVFP4, 130B+, conc-32]
status: done
prefill_toks: 15.74
decode_toks: 11.02
mem_gb: 98.6
mem_source: ds4 managed-model resident (98.64 GiB, engine log); 32K KV is negligible; system `free -h` showed 101 GiB used incl ~3.4 GiB OS baseline
measured_on: 2026-06-28
completed_at: 2026-06-28 13:34 +08
engine_image: ds4-flash:local — built from github.com/sleepyeldrazi/ds4-nvfp4-spark @ 2060b3def7d7555b3e0a6c5b0cc374ba49702c96, base nvidia/cuda:13.0.1-devel-ubuntu24.04, `make CUDA_ARCH=sm_121a`
run_command: |
  # Same custom ds4-flash:local image as the conc-1 page (build recipe there). Model needs ~99 GiB
  # resident — cannot coexist with any other engine on the 121 GB box.
  docker run -d --name serving-ds4-flash --gpus all \
    -e DS4_CUDA_MANAGED_MODEL=1 -e DS4_KV_TURBO=1 \
    -v ~/ds4flash/gguf:/models:ro -p 127.0.0.1:8000:8000 \
    ds4-flash:local \
    -m /models/DeepSeek-V4-Flash-REAP-K180-hybrid.gguf --host 0.0.0.0 --port 8000 --ctx 32768
  python3 scripts/bench-serving.py --base-url http://127.0.0.1:8000 --model deepseek-v4-flash \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Concurrency buys nothing on this runtime — `ds4-server` is a serial single-worker, so conc-32 just builds
a giant queue at the same ~11 tok/s aggregate.** Author's REAP-K180 hybrid GGUF on `ds4-nvfp4-spark` (GB10,
sm_121a), ctx 32768, conc 32. Pair with the **conc-1 (1M ctx) sibling** for the usable single-stream point.

- **Load:** ready in **100 s** via CUDA managed memory; 105.9 GB GGUF → **98.64 GiB resident**. 32K KV is
  negligible, so total ≈ 101 GiB used (incl OS).
- **The runtime processes one request at a time.** The live engine log shows strictly sequential prompts
  (`ctx=0..10`, then `0..9`, then `0..55` …) at a steady ~12.3 t/s decode — there is **no continuous
  batching**. So driving 32 concurrent requests does not raise throughput; it only deepens the queue.
- **Result of that:** decode **11.02 tok/s aggregate** (≈ the same ~12.4 tok/s/req single-stream rate, TPOT
  median 80.5 ms), prefill 15.74 tok/s — but **TTFT median 576 s** (≈ 9.6 min) and **9 client-side timeouts**
  as requests pile up. **68/1000 completed, 9 errors**, 900 s cap (`hit_time_cap=true`).
- **req throughput 0.046 req/s — identical to the conc-1 run.** Confirms the ceiling is the serial decoder,
  not concurrency.
- **Operational takeaway:** drive this model at **conc 1**. Same tokens/sec, sub-second TTFT, zero errors.
  Concurrency on a serial backend is pure tail-latency damage.
- **No speculative path** (serial single worker; DeepSeek MTP head unused) — same as the conc-1 page.
