---
title: Qwen3.6-27B · vLLM · FP8
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3.6-27B-FP8). Qwen3.6-27B is the current flagship dense Qwen — per Qwen, it beats the 397B Qwen3.5 model on coding benchmarks. The novel model the model list's "qwen3-6-27b" stub actually meant (earlier mis-recovered to Qwen3-32B; that valid run is now qwen3-32b-vllm-fp8).
source_repo: Qwen/Qwen3.6-27B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B, conc-32]
status: done
prefill_toks: 168.88
decode_toks: 154.66
mem_gb: 107.45
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 18:11 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). qwen3_5 arch loads on the stock image.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-27B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The current flagship dense Qwen — and a 27B that reportedly beats the Qwen3.5-397B on coding.**
Qwen3.6-27B, official FP8 on vLLM. Replaced the pending Qwen3.5-397B-A17B in the queue (Qwen3.6 has no
giants; the 27B dense wins on benchmarks).

- **Arch note (good news):** the very new `qwen3_5` / `Qwen3_5ForConditionalGeneration` (multimodal)
  arch **loads on the stock cu130-nightly vLLM** — no transformers bump needed (unlike Gemma's
  `gemma4_unified`). Benchmarked text-only (`mm_served: false`).
- **Workload:** ShareGPT V3, concurrency 32. **576/1000, 0 errors** — clean, but **hit the 15-min cap**.
- **Throughput (aggregate, conc 32):** prefill **168.9 tok/s**, decode **154.7 tok/s**. TTFT median
  ~clean, req throughput ~0.6/s.
- **A dense-27B that performs like the dense-32B tier.** Decode **154.7** sits right with the older
  Qwen3-32B FP8 (156) — same dense-transformer bandwidth wall on the GB10, with the 27B's slightly
  smaller weight footprint roughly offset by being benchmarked at the same ctx. The 3.6 generation's
  gains are in *quality* (it's a 27B beating a 397B), not raw decode speed, which is architecture- and
  size-bound. The **native-MTP** variant (`+ MTP`) is where its decode throughput can actually move.
- **Memory: 107.5 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint (FP8
  weights ≈ 27 GB).
