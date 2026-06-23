---
title: Gemma 4 12B · vLLM · NVFP4 · RedHatAI
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: vLLM
quant: NVFP4
quant_rationale: RedHatAI/gemma-4-12B-it-NVFP4 — trusted Red Hat AI NVFP4 quant, with the model card explicitly stating it is compatible with and tested against vLLM nightly. Queue it on vLLM rather than SGLang because that is the validated serving stack the repo names directly.
source_repo: RedHatAI/gemma-4-12B-it-NVFP4
download_url: https://huggingface.co/RedHatAI/gemma-4-12B-it-NVFP4
context: 65536
modalities: [text, image, audio, video]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, NVFP4, 5-15B, conc-32]
status: done
prefill_toks: 582.39
decode_toks: 503.81
mem_gb: 109.38
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85)
measured_on: 2026-06-23
completed_at: 2026-06-23 23:07 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # RedHatAI says this checkpoint is compatible with and tested against vLLM nightly.
  scripts/bench-vllm-serving.sh RedHatAI/gemma-4-12B-it-NVFP4 65536 32 1000 900 256
---

**DONE — the smallest pending benchmark is now recorded, and this Red Hat AI NVFP4 path runs cleanly on
the maintained vLLM nightly.**

- **Workload:** ShareGPT V3, conc-32, standard 1000-prompt / 15 min cap. This run completed
  **1000/1000 with 0 errors** in **462.7 s** and did **not** hit the time cap.
- **Throughput:** prefill **582.39 tok/s**, decode **503.81 tok/s** aggregate. Median **TTFT 179.3 ms**,
  median **TPOT 58.3 ms**, request throughput **2.161 req/s**.
- **Load behavior:** server readiness took **226 s** end to end before the benchmark began. Once up, it
  stayed stable under the full 32-way load.
- **Memory:** **109.38 GB** headline usage, which here is the expected **vLLM KV reservation** at
  `--gpu-memory-utilization 0.85`, not the checkpoint footprint. The engine log reported the model load
  itself at about **8.29 GiB** before KV allocation/cudagraph overhead.
- **Engine/runtime notes:** vLLM resolved the checkpoint as **`Gemma4UnifiedForConditionalGeneration`**
  on `nightly-aarch64`, forced the **TRITON_ATTN** backend because Gemma-4 has heterogeneous head
  dimensions, and used the **FlashInferCutlassNvFp4LinearKernel** NVFP4 GEMM path. A runtime warning
  flagged **different global NVFP4 scales across some parallel layers**, which may slightly reduce
  accuracy; throughput was unaffected in this run.
