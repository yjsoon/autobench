---
title: gpt-oss-20b · vLLM · MXFP4 + EAGLE3 · conc 8
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
concurrency: 8
tags: [gpt-oss-20b, OpenAI, gpt-oss, MXFP4, 16-40B, Spark recipe, conc-8]
status: done
prefill_toks: 144.02
decode_toks: 126.3
mem_gb: 108.10
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + EAGLE3 head
measured_on: 2026-06-22
completed_at: 2026-06-22 23:13 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly, harmony vocab pre-seeded, RedHatAI EAGLE3 speculator.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v ~/models/tiktoken_cache:/vocab:ro \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/vocab \
    vllm/vllm-openai:cu130-nightly openai/gpt-oss-20b \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 8 \
    --speculative-config '{"model":"RedHatAI/gpt-oss-20b-speculator.eagle3","method":"eagle3","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model openai/gpt-oss-20b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 8 --max-tokens 256
---

**Mid-concurrency point of the EAGLE3 sweep — acceptance degrades *further* with batch, the opposite of
the "constant-with-concurrency" rule of thumb.** gpt-oss-20b MXFP4 + RedHatAI EAGLE3 speculator on vLLM,
conc 8.

- **Workload:** ShareGPT V3, concurrency 8. **153/500 completed, 10 errors** before the **300 s time cap**
  (`hit_time_cap=true`).
- **Throughput:** prefill **144.02 tok/s**, decode **126.3 tok/s** aggregate (~15.8 tok/s/stream).
- **EAGLE3 acceptance: even worse than conc-1 — a red flag.** Across the run: **mean acceptance length
  ~1.1–1.36 (centered ~1.2)**, **avg draft acceptance ~3–12% (centered ~5–8%)**, per-position only
  **~0.05–0.21 / 0.02–0.11 / 0.01–0.05**. Compare conc-1 (mean ~1.7, ~25%): acceptance **dropped as
  concurrency rose**. Per CLAUDE.md acceptance is supposed to be **workload-driven, not concurrency-driven**
  — this downward swing is a genuine deviation. Likely cause: at higher batch vLLM's spec scheduler verifies
  draft tokens under more contention and the already-weak draft (ShareGPT + harmony reasoning channel)
  converts even less; the draft is essentially **dead weight** here (drafting ~400 tok/s, accepting ~25).
- **Harmony corruption worse at batch (10 errors).** Beyond the conc-1 single error, conc-8 produced
  **garbled harmony** mid-stream — e.g. `Unknown role: assistant<|end|>!!!<|end|><|start|>!!<|end|>!…`
  and `Unexpected token 0 while expecting start token 200006`. The EAGLE3 draft appears to emit
  off-distribution tokens that corrupt the harmony channel structure, which the parser then rejects. This
  contradicts the [conc-32 page]'s "EAGLE3 fixed the harmony errors" claim — error-elimination is **not**
  reliable; it's stochastic and *worsens* at conc 8.
- **Harmony artifacts:** TTFT median **13.3 s** (buffered reasoning + 8-way queue) and TPOT **0.0** are not
  real latencies; aggregate tok/s are the valid headline.
- **Memory: 108.1 GB** = vLLM 0.85 reservation + EAGLE3 head, not footprint.
- **Verdict:** EAGLE3 on gpt-oss-20b is a poor fit on this workload at low/mid concurrency — low,
  concurrency-degrading acceptance plus harmony-corruption errors. The [conc-32] win (+28% decode, 0
  errors) does **not** generalize down the concurrency sweep; treat that result as concurrency-specific.
