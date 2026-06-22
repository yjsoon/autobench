---
title: Gemma 4 31B · vLLM · FP8
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: vLLM
quant: FP8
quant_rationale: RedHatAI official FP8-Dynamic (compressed-tensors) — the 1-byte point on the 31B quant ladder, to isolate quant effect from engine vs the NVFP4 run.
source_repo: RedHatAI/gemma-4-31B-it-FP8-Dynamic
download_url: https://huggingface.co/RedHatAI/gemma-4-31B-it-FP8-Dynamic
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-31b, Google, Gemma, FP8, 16-40B, conc-32]
status: done
prefill_toks: 174.77
decode_toks: 147.62
mem_gb: 106.29
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 10:10 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). RedHatAI FP8-Dynamic.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly RedHatAI/gemma-4-31B-it-FP8-Dynamic \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model RedHatAI/gemma-4-31B-it-FP8-Dynamic \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**FP8 on the dense 31B — and the quant tax is much smaller here than on the MoE, because this model is
compute-bound, not bandwidth-bound.** Google's Gemma-4-31B, RedHatAI FP8-Dynamic on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **603/1000, 0 errors** — **hit the 15-min cap** (the dense
  31B stays expensive even at FP8). Loaded in **416 s**.
- **Throughput (aggregate, conc 32):** prefill **174.8 tok/s**, decode **147.6 tok/s**. TTFT median
  **617 ms**, TPOT median **196 ms** (≈5.1 tok/s/stream), req throughput 0.64/s.
- **The dense-31B ladder, and why quant moves it less than the 26B MoE:**

  | Quant / engine | decode | completed | note |
  |---|---|---|---|
  | Q4_K_M / llama.cpp | 78.5 | 322/1000 | KV-cliff disaster |
  | **FP8 / vLLM** | **147.6** | 603/1000 | |
  | NVFP4 / vLLM | 167.0 | 672/1000 | best |

  On vLLM, **FP8 → NVFP4 is only +13%** (148 → 167) for this dense model, vs **+21%** and an overall
  **2×** on the 26B *MoE*. The reason: a dense 33B fires **all** its parameters every token, so decode is
  **compute-bound** — halving the weight bytes (FP8→NVFP4) helps the memory side but the matmul FLOPs
  don't shrink, so the gain is modest. The sparse MoE activates only ~4B/token, making it
  weight-**bandwidth**-bound, where the same quant step pays off much more. **Quant format helps
  bandwidth-bound models far more than compute-bound ones** — this pair is the clean evidence.
- **Memory: 106.3 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (FP8 weights ≈ 33 GB).
- Text path benchmarked (`mm_served: false`).
