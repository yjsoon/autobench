---
title: Gemma 4 31B · vLLM · NVFP4 + EAGLE3 · conc 8
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: vLLM
speculative: EAGLE3
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + RedHatAI's official EAGLE3 speculator — fast quant plus lossless speculative decoding, the combination that finally clears the 31B's time cap.
source_repo: nvidia/Gemma-4-31B-IT-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 8
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B, conc-8]
status: done
prefill_toks: 119.32
decode_toks: 91.47
mem_gb: 107.17
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-23
completed_at: 2026-06-23 01:13 +08
engine_image: vllm/vllm-openai:cu130-nightly@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
run_command: |
  # vllm/vllm-openai:cu130-nightly. NVFP4 base + RedHatAI EAGLE3 speculator (draft model).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-31B-IT-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-31B-IT-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Conc-8 point for the Gemma 4 31B EAGLE3 sweep — acceptance steady at ~2.3, 0 errors. Closes the Gemma
vLLM NVFP4+EAGLE3 set.** NVIDIA NVFP4 base + RedHatAI EAGLE3 speculator on vLLM (cu130-nightly),
ctx 65536, conc 8.

- **Load:** ready in **392 s**.
- **Workload:** ShareGPT V3, concurrency 8. **129/500 completed, 0 errors** before the **300 s time cap**.
- **Throughput:** prefill **119.32 tok/s**, decode **91.47 tok/s** aggregate (~11.4 tok/s/stream). TTFT
  median **562 ms**, TPOT median **79.9 ms**.
- **EAGLE3 acceptance — steady ~2.3.** Run-aggregate **mean acceptance length ~2.18–2.57 (centered ~2.3)**,
  **avg draft acceptance ~39–52% (centered ~44%)**, per-position **~0.61–0.74 / 0.35–0.50 / 0.21–0.33** —
  essentially unchanged from conc-1 (~2.4), confirming acceptance is workload-driven. Highest vLLM-EAGLE3
  acceptance of the two Gemma models (vs 26b-a4b ~2.15), matching the "bigger dense model drafts better"
  trend seen in the MTP runs.
- **Memory: 107.2 GB** = vLLM 0.85 reservation + EAGLE3 head, not footprint.
- **Closes the Gemma vLLM NVFP4+EAGLE3 sweep** (26b-a4b + 31b × conc-1/8): all clean (0 errors), EAGLE3 a
  solid ~2.0–2.4× draft on ShareGPT, stable across concurrency — and all on **cu130-nightly**, since the
  newer nightly-aarch64 (0.23.1) can't load Gemma-4 NVFP4 (tie_weights regression; see the conc-1 page).
