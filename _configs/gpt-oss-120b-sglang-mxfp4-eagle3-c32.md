---
title: gpt-oss-120b · SGLang · MXFP4 + EAGLE3 · conc 32
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: SGLang
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss native MXFP4 base + SGLang's own LMSYS/SpecForge EAGLE3 draft (lmsys/EAGLE3-gpt-oss-120b-bf16). conc-32 throughput point — directly comparable to base SGLang (decode 140 tok/s) and vLLM+EAGLE3 (138). Tests whether the conc-1 spec-decode win survives heavy batching on a 120B model.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-32]
status: done
prefill_toks: 213.35
decode_toks: 171.86
mem_gb: 113.45
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation + EAGLE3 draft
measured_on: 2026-06-23
completed_at: 2026-06-23 21:20 +0800
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # REQUIRED the newer nightly image — the spark image drops ~70% of conc-32 streams with EAGLE3
  # (Connection reset; see INCOMPATIBILITIES.md). Override: SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed
  docker run -d --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/tiktoken_encodings:/tiktoken_encodings \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings \
    lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b --host 0.0.0.0 --port 30000 \
    --context-length 65536 --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  python3 scripts/bench-serving.py --base-url http://localhost:30000 \
    --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The throughput regime — does the conc-1 EAGLE3 win survive 32-way batching?** SGLang's LMSYS/SpecForge
draft (`lmsys/EAGLE3-gpt-oss-120b-bf16`) on gpt-oss-120b, conc 32. Companion to the
[conc-1] run (decode 40.56 tok/s, accept-len ~2.4) — compare against base SGLang (decode **140.3**) and
vLLM+EAGLE3 (decode **138.5**, which *lost* ~45% vs its base at this concurrency).

- **The win survives heavy batching — EAGLE3 is net-positive at conc-32 too.** decode **171.86 tok/s**
  aggregate vs base SGLang's **140.3** = **+22%**, with **0 errors** (652/1000, hit the 900 s cap).
  accept-len held **~2.25 median** (2.01–2.45) — essentially flat from conc-1's ~2.4, exactly the
  "acceptance is workload-driven, not concurrency-driven" rule. This is the **opposite** of vLLM+EAGLE3
  here (decode 138.5, −45% vs its base) and of the NVIDIA-draft collapse (accept ~1.05 at conc-8/32).
  Prefill 213.35 tok/s; TTFT median 1.21 s, TPOT median 175.7 ms.
- **⚠️ Required the newer SGLang nightly — the `spark` image is broken for this path.** On
  `lmsysorg/sglang:spark`, conc-32 + EAGLE3 **reproducibly dropped ~70% of requests** (`Connection reset
  by peer`; 298/1000 completed, 702 errors — confirmed across 3 runs, including with the box otherwise
  idle). It is **not** OOM, CUDA, harmony corruption, or the draft (server logs clean, single requests
  fine, acceptance healthy ~2.2). EAGLE3 disables SGLang's overlap scheduler, and the spark image's
  serving path can't sustain 32 concurrent streams in that mode. **`nightly-dev-cu13-20260623-ba9d5aed`
  fixes it: 0 errors.** Full write-up in `INCOMPATIBILITIES.md`.
- **Image caveat on the +22%:** the base 140.3 was measured on the `spark` image; this EAGLE3 run is on
  the nightly. The speedup is therefore *approximate* (mixed images) — but the direction is unambiguous
  (0 errors, accept ~2.2, decode well above base), and the conc-1 same-engine result corroborates the
  win. A base-on-nightly re-run would tighten the exact percentage.
- **Memory 113.45 GB** = SGLang static reservation + the 0.9B draft, a reservation not the footprint.
- **Takeaway:** for gpt-oss-120b *throughput* on one Spark, SGLang + the LMSYS EAGLE3 draft on a current
  nightly **beats base** (172 vs 140) and beats every other spec path measured — the first spec-decode
  win on gpt-oss-120b in this benchmark. Pair with the [conc-1] latency win (40.6 tok/s).
