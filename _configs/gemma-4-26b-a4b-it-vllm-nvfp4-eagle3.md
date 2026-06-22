---
title: Gemma 4 26B-A4B · vLLM · NVFP4 + EAGLE3
model: google/gemma-4-26B-A4B-it
company: Google
family: Gemma
params: 26B / 4B (MoE)
engine: vLLM
speculative: EAGLE3
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + RedHatAI's official EAGLE3 speculator — the fast quant plus lossless speculative decoding stacked together.
source_repo: nvidia/Gemma-4-26B-A4B-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-26b-a4b, Google, Gemma, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 619.97
decode_toks: 540.99
mem_gb: 109.41
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 10:29 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly. NVFP4 base + RedHatAI EAGLE3 speculator (draft model).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-26B-A4B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"model":"RedHatAI/gemma-4-26B-A4B-it-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-26B-A4B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The fastest Gemma config measured — NVFP4 weights and EAGLE3 speculation stacked, +41% decode over
the already-fast NVFP4 base.** Google's Gemma-4-26B-A4B (26B / 4B active), NVIDIA NVFP4 base +
RedHatAI's official EAGLE3 speculator, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **435 s** — the quickest full
  1000-prompt completion of any Gemma run. Loaded + CUDA-graph captured in **273 s**.
- **Throughput (aggregate, conc 32):** prefill **620.0 tok/s**, decode **541.0 tok/s**. TTFT median
  **336 ms**, TPOT median **54.1 ms** (≈18.5 tok/s/stream), req throughput 2.3/s.
- **EAGLE3 speedup over the NVFP4 base (same model/engine/quant, speculation on vs off):**

  | Config | decode | TPOT med | duration | req/s |
  |---|---|---|---|---|
  | NVFP4 (base) | 384.1 | 79.3 ms | 613 s | 1.63 |
  | **NVFP4 + EAGLE3** | **541.0** | **54.1 ms** | **435 s** | **2.30** |

  **~1.41× decode, 1.47× lower per-token latency, 1.41× request throughput** — and lossless (EAGLE3 only
  accepts draft tokens the target would have produced). The draft head proposes 3 tokens/step; on
  ShareGPT's fairly predictable chat continuations the acceptance rate is high enough to convert most
  steps into multi-token emits. This is the **fastest Gemma decode in the benchmark**, ~2.8× the BF16
  base (190) it started from.
- **Spec-decode engine note — this is the *only* working spec-decode path for Gemma 4 on the stock
  images.** SGLang's spark image has no `gemma4` model support (transformers 4.57.1), so the
  SGLang-NEXTN / Google-assistant-drafter route is blocked without a custom fork. vLLM EAGLE3 with the
  RedHatAI speculator works here on the cu130-nightly (0.19.2) build — the multimodal draft-model
  restriction noted in vLLM issue #42005 (0.20.1) did **not** trigger on this older build.
- **Memory: 109.4 GB is the vLLM `--gpu-memory-utilization 0.85` reservation** (NVFP4 weights ≈ 14 GB +
  the small EAGLE3 head), not the footprint.
- Text path benchmarked (`mm_served: false`).
