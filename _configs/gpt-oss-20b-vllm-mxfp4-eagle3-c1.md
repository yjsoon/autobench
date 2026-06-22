---
title: gpt-oss-20b · vLLM · MXFP4 + EAGLE3 · conc 1
model: openai/gpt-oss-20b
company: OpenAI
family: gpt-oss
params: 21B / 3.6B (MoE)
engine: vLLM
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss MXFP4 base + RedHatAI's EAGLE3 speculator (speculators format, 32k dl/mo) — the spec-decode dimension on gpt-oss-20b, pairing with its SGLang/vLLM/llama.cpp configs.
source_repo: openai/gpt-oss-20b
download_url: https://huggingface.co/openai/gpt-oss-20b
context: 65536
modalities: [text]
mm_served: true
concurrency: 1
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe, conc-1]
status: done
prefill_toks: 66.09
decode_toks: 38.55
mem_gb: 108.03
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-22
completed_at: 2026-06-22 23:03 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, RedHatAI EAGLE3 speculator.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 1 \
    --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-20b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**Single-stream point of the EAGLE3 concurrency sweep — and the run that finally surfaces the (low)
acceptance rate.** gpt-oss-20b MXFP4 + RedHatAI EAGLE3 speculator on vLLM, conc 1.

- **Workload:** ShareGPT V3, concurrency 1. **49/500 completed, 1 error** before the **300 s time cap**
  (`hit_time_cap=true`) — at conc-1 the serial throughput naturally caps the count well under 500; this
  is a latency characterization, not a throughput-to-N run.
- **Throughput:** prefill **66.09 tok/s**, decode **38.55 tok/s** — these are *single-stream* rates, so
  decode is ~1.8× the per-stream share of the [conc-32 run] (686.5/32 ≈ 21 tok/s/stream): less batch
  contention at conc 1, as expected.
- **EAGLE3 acceptance (NEW — the conc-32 page didn't capture it): poor on this workload.** Across the run
  the vLLM SpecDecoding metrics swung **mean acceptance length ~1.0–2.4 (centered ~1.7)**, **avg draft
  acceptance ~10–46% (centered ~22–30%)**, per-position roughly **0.3–0.5 / 0.15–0.35 / 0.05–0.25** for
  the 3 draft slots. That's **well below EAGLE3's expected ~3.0 mean / ~70%+** — i.e. with
  `num_speculative_tokens=3` the draft lands barely over one extra token on average. **Red-flag check:**
  per CLAUDE.md this gap is *workload-driven, not a misconfig* — ShareGPT general chat + gpt-oss's
  **harmony reasoning ("analysis") channel** is hard to draft (the speculator was trained/tuned for
  different output), and CLAUDE.md already notes ShareGPT runs lower than code. Acceptance stays
  workload-bound, so expect similarly low numbers at conc 8/32.
- **Harmony artifacts (same as conc-32):** TTFT median **4.6 s** and TPOT median **0.0** are the
  buffered-reasoning chat-path artifacts, not real latencies — the aggregate tok/s above are the valid
  headline. 1 request also tripped the harmony parser bug (`unexpected tokens remaining in message
  header`), counted as the single error.
- **Memory: 108.0 GB** is the vLLM `--gpu-memory-utilization 0.85` reservation + EAGLE3 head, not the
  footprint (MXFP4 weights ≈ 11 GB).
- **Takeaway:** low acceptance means the conc-1 decode gain over a (hypothetical) base is modest despite
  max headroom — the draft simply isn't accurate enough on this workload to convert headroom into
  speedup. The +28% the conc-32 page saw came mostly from prefill/scheduling, not high acceptance.
