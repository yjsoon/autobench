---
title: Gemma 4 26B-A4B · vLLM · FP8
model: google/gemma-4-26B-A4B-it
company: Google
family: Gemma
params: 26B / 4B (MoE)
engine: vLLM
quant: FP8
quant_rationale: RedHatAI official FP8-Dynamic (compressed-tensors) — the 1-byte middle point between BF16 and NVFP4 for the quant-tax ladder.
source_repo: RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
download_url: https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic
context: 65536
modalities: [text, image]
mm_served: false
tags: [gemma-4-26b-a4b, Google, Gemma, FP8, 16-40B]
status: done
prefill_toks: 362.64
decode_toks: 316.93
mem_gb: 98.66
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 09:47 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). RedHatAI FP8-Dynamic.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model RedHatAI/gemma-4-26B-A4B-it-FP8-Dynamic \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The middle rung of the Gemma-26B quant ladder — FP8 lands exactly between BF16 and NVFP4.**
Google's Gemma-4-26B-A4B (26B / 4B active), RedHatAI FP8-Dynamic on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **743 s** — clean full run, no
  time cap. Loaded in **356 s**.
- **Throughput (aggregate, conc 32):** prefill **362.6 tok/s**, decode **316.9 tok/s**. TTFT median
  **311 ms**, TPOT median **97.6 ms** (≈10 tok/s/stream), req throughput 1.35/s.
- **The full quant-tax ladder for one model, and it tracks bytes-per-weight almost linearly:**

  | Quant | bytes/weight | prefill | decode | run |
  |---|---|---|---|---|
  | BF16 | 2.0 | 212.7 | 190.1 | time-capped |
  | **FP8** | **1.0** | **362.6** | **316.9** | clean |
  | NVFP4 | 0.5 | 439.3 | 384.1 | clean |

  Decode rises **190 → 317 → 384** as the weight format halves twice. The BF16→FP8 step (+67%) is bigger
  than FP8→NVFP4 (+21%) — diminishing returns, because at NVFP4 the run is no longer purely
  weight-bandwidth-bound (KV traffic and compute start to matter). Still, FP8 alone is enough to clear
  the time cap and pull the model into the fast tier; NVFP4 is the extra margin.
- **Memory: 98.7 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (FP8 weights ≈ 27 GB). Note it's a touch lower than the NVFP4 run's reservation — vLLM sized the KV
  pool slightly differently, not a real footprint signal.
- Text path benchmarked (`mm_served: false`).
