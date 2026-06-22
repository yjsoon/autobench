---
title: Qwen3.6-35B-A3B · vLLM · FP8
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
quant: FP8
quant_rationale: Official FP8 weights (Qwen/Qwen3.6-35B-A3B-FP8). The sparse-MoE 3.6 model — 3B active, agentic-coding focused; per Qwen it dramatically beats its 3.5 predecessor and is the natural replacement for the Qwen3.5-122B-A10B MoE.
source_repo: Qwen/Qwen3.6-35B-A3B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-35B-A3B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, FP8, 16-40B, conc-32]
status: done
prefill_toks: 294.57
decode_toks: 285.97
mem_gb: 107.88
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 19:09 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). qwen3_5_moe loads on stock image.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-35B-A3B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-35B-A3B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The sparse-MoE Qwen3.6 — fast 3B-active decode, clean 1000/1000.** Qwen3.6-35B-A3B (35B total / 3B
active), official FP8 on vLLM. Replaced the blocked Qwen3.5-122B-A10B MoE in the queue.

- **Arch:** `qwen3_5_moe` (multimodal) loads on the stock cu130-nightly vLLM — no transformers bump
  needed. Benchmarked text-only (`mm_served: false`).
- **Workload:** ShareGPT V3, concurrency 32. **1000/1000, 0 errors** in **895 s** — clean full run, no
  time cap (unlike the dense 27B, which capped).
- **Throughput (aggregate, conc 32):** prefill **294.6 tok/s**, decode **286.0 tok/s** — **1.85× the
  dense Qwen3.6-27B's decode (154.7)**, the classic MoE-vs-dense gap (3B active vs 27B dense). Lands
  right with the other 3B-active FP8 MoEs (Qwen3-30B-A3B 331, Qwen3-Coder-30B-A3B 296). TTFT/TPOT
  clean, req throughput 1.1/s.
- **The 3.6 generation's edge is quality, not speed** — this MoE decodes like its Qwen3 A3B cousins;
  its value is matching/beating much larger 3.5 models on coding at this speed. The native-MTP variant
  (`+ MTP`) tests whether speculation lifts it further (it has compute headroom at 3B active).
- **Memory: 107.9 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint (FP8
  weights ≈ 35 GB).
