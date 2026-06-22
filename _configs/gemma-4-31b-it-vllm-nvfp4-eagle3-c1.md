---
title: Gemma 4 31B · vLLM · NVFP4 + EAGLE3 · conc 1
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
concurrency: 1
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B, conc-1]
status: done
prefill_toks: 3.37
decode_toks: 14.78
mem_gb: 108.16
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-23
completed_at: 2026-06-23 01:01 +08
engine_image: vllm/vllm-openai:cu130-nightly@sha256:3dbe092ec5b2cef63b6104d33fa75d6ce53a7870962529ada69f78bbbc38e776
run_command: |
  # vllm/vllm-openai:cu130-nightly. NVFP4 base + RedHatAI EAGLE3 speculator (draft model).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-31B-IT-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"model":"RedHatAI/gemma-4-31B-it-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-31B-IT-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Conc-1 point for the Gemma 4 31B EAGLE3 sweep — highest vLLM-EAGLE3 acceptance in the Gemma set
(~2.4 accept-len, ~47% draft accept), tracking the "bigger dense model drafts better" trend.** NVIDIA
NVFP4 base + RedHatAI EAGLE3 speculator on vLLM (cu130-nightly), ctx 65536, conc 1.

- **Load:** ready in **404 s** (~6.7 min) — the dense 31B NVFP4 + EAGLE3 head + CUDA-graph capture is the
  slowest Gemma vLLM load.
- **Workload:** ShareGPT V3, concurrency 1. **19/500 completed, 0 errors** before the **300 s time cap** —
  the 31B is the slowest Gemma, so only ~19 single-stream requests finish in 5 min.
- **Throughput:** decode **14.78 tok/s** (single stream), TPOT median **68.1 ms**. (Prefill **3.37 tok/s**
  is an artifact of the tiny completed count + short prompts — not a real compute rate.)
- **EAGLE3 acceptance — best vLLM-EAGLE3 in the set, ~2.4.** Across the run: **mean acceptance length
  ~2.26–2.61 (centered ~2.4)**, **avg draft acceptance ~42–54% (centered ~47%)**, per-position
  **~0.62–0.78 / 0.40–0.53 / 0.20–0.33**. Higher than the 26b-a4b MoE EAGLE3 (~2.0 / ~33%) — same pattern
  as the MTP runs, where the bigger *dense* model self-drafts best (cf. 31B MTP accept-len 3.41). Still
  below MTP because EAGLE3 is a separate small head, not the model's own MTP layer. 0 errors.
- **Memory: 108.2 GB** = vLLM 0.85 reservation + EAGLE3 head, not footprint.
- Compare decode + TPOT against the conc-32 run: at conc-1 the per-stream decode is 14.78 with the highest
  acceptance; conc-32 trades per-stream latency for aggregate throughput while acceptance stays ~constant.
