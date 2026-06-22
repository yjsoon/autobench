---
title: Gemma 4 26B-A4B · vLLM · NVFP4
model: google/gemma-4-26B-A4B-it
company: Google
family: Gemma
params: 26B / 4B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: NVIDIA's own NVFP4 build (TensorRT-Model-Optimizer / modelopt) — Blackwell-native 4-bit. The headliner swap to lift the 26B MoE out of the BF16 slow tier.
source_repo: nvidia/Gemma-4-26B-A4B-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-26b-a4b, Google, Gemma, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 439.25
decode_toks: 384.12
mem_gb: 109.50
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 09:27 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). NVIDIA NVFP4 (modelopt).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-26B-A4B-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-26B-A4B-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**NVFP4 doubles the BF16 MoE and vaults it into the fast tier — the cleanest quant-tax result in the
benchmark.** Google's Gemma-4-26B-A4B (26B / 4B active) in NVIDIA's NVFP4 build, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **613 s** — a clean full run,
  **no time cap** (the BF16 build of the same model *did* hit the cap). Loaded in **284 s**.
- **Throughput (aggregate, conc 32):** prefill **439.3 tok/s**, decode **384.1 tok/s**. TTFT median
  **249 ms**, TPOT median **79.3 ms** (≈13 tok/s/stream), req throughput 1.63/s.
- **Same model, NVFP4 vs BF16 — a clean ~2× on both axes:**

  | Quant | prefill | decode | completed | time-cap |
  |---|---|---|---|---|
  | BF16 (base) | 212.7 | 190.1 | 745/1000 | yes |
  | **NVFP4** | **439.3** | **384.1** | **1000/1000** | no |

  A 4B-active MoE "should" be fast, and BF16 hid that behind 2× the per-weight byte traffic on
  bandwidth-bound decode. Drop to Blackwell-native NVFP4 and it lands at **384 decode — right in the
  NVFP4 Nemotron-30B-A3B league (353–389)**, exactly where a sparse-active MoE belongs. This is the
  single clearest demonstration in the whole set that **quant format, not architecture, was the
  bottleneck** for these Gemmas.
- **Memory: 109.5 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (NVFP4 weights ≈ 14 GB).
- Text path benchmarked (`mm_served: false`).
