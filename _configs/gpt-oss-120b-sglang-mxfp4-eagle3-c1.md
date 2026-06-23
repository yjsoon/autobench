---
title: gpt-oss-120b · SGLang · MXFP4 + EAGLE3 · conc 1
model: openai/gpt-oss-120b
company: OpenAI
family: gpt-oss
params: 116.8B (5.1B active, MoE)
engine: SGLang
speculative: EAGLE3
quant: MXFP4
quant_rationale: gpt-oss native MXFP4 base + SGLang's own LMSYS/SpecForge EAGLE3 draft (lmsys/EAGLE3-gpt-oss-120b-bf16, 0.9B BF16) — the SGLang-canonical spec-decode path. Distinct draft from the vLLM configs (which used nvidia/gpt-oss-120b-Eagle3-throughput); this tests whether a SGLang-native draft escapes the off-distribution acceptance collapse seen on vLLM.
source_repo: openai/gpt-oss-120b
download_url: https://huggingface.co/openai/gpt-oss-120b
context: 65536
modalities: [text]
mm_served: true
concurrency: 1
tags: [gpt-oss-120b, OpenAI, gpt-oss, MXFP4, 41-130B, Spark recipe, conc-1]
status: done
prefill_toks: 52.09
decode_toks: 40.56
mem_gb: 114.08
mem_source: system MemAvailable delta (10s sampling) — SGLang static KV reservation + EAGLE3 draft, see Notes
measured_on: 2026-06-23
completed_at: 2026-06-23 20:08 +0800
engine_image: lmsysorg/sglang:spark@sha256:16dec654b13e4d10a2d9eefa0560e85fed0d1fc9536986e1dfb1bcb0077cbc7a
run_command: |
  # lmsysorg/sglang:spark; harmony tiktoken encodings mounted; LMSYS/SpecForge EAGLE3 draft.
  docker run -d --gpus all --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v ~/tiktoken_encodings:/tiktoken_encodings \
    --env HF_TOKEN=*** --env TIKTOKEN_ENCODINGS_BASE=/tiktoken_encodings \
    lmsysorg/sglang:spark python3 -m sglang.launch_server \
    --model-path openai/gpt-oss-120b --host 0.0.0.0 --port 30000 \
    --context-length 65536 --reasoning-parser gpt-oss --tool-call-parser gpt-oss \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16 \
    --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  python3 scripts/bench-serving.py --base-url http://localhost:30000 \
    --model openai/gpt-oss-120b \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 1 --max-tokens 256
---

**SGLang-native EAGLE3 on the gpt-oss-120b headliner, single-stream.** Uses the LMSYS/SpecForge draft
`lmsys/EAGLE3-gpt-oss-120b-bf16` (0.9B, BF16, 1.77 GB) — the draft SGLang's own gpt-oss recipe
([SGLang docs](https://docs.sglang.io/advanced_features/speculative_decoding.html),
[LMSYS gpt-oss blog](https://www.lmsys.org/blog/2025-08-27-gpt-oss/)) ships, **distinct** from the
`nvidia/gpt-oss-120b-Eagle3-throughput` head the vLLM EAGLE3 configs used. conc-1 is the single-stream
latency regime EAGLE3 is built for — and the cleanest acceptance read.

- **The headline: the SGLang-native LMSYS draft *works* — unlike the NVIDIA draft on vLLM.** mean
  **accept len ~2.4** (range ~1.75–2.85), **accept rate ~0.60** (~0.44–0.71), holding steady across the
  whole single-stream run. That is right in the healthy EAGLE3 band — a complete reversal of the vLLM
  EAGLE3 conc-1 result (`nvidia/gpt-oss-120b-Eagle3-throughput`), which collapsed to **accept-len ~1.25 /
  ~9%** on the same ShareGPT+harmony workload. **Same base model, same workload, same concurrency — the
  draft is what changed.** The LMSYS/SpecForge head is in-distribution for general chat; the NVIDIA
  throughput head is not.
- **Throughput (single stream, conc 1):** decode **40.56 tok/s**, prefill 52.09 tok/s. TTFT median
  **160.6 ms**, TPOT median **21.1 ms**. The live `gen throughput` ran **~40–56 tok/s**. Versus the vLLM
  EAGLE3 conc-1 point (decode **14.69 tok/s**), this is **~2.8×** — though note the engines differ;
  see the conc-32 sibling for the same-engine base-vs-spec delta.
- **Zero errors / no harmony corruption.** 153/1000 completed, **0 errors** before the 900 s cap (single
  stream is slow — `hit_time_cap=true`). The vLLM EAGLE3 runs suffered harmony-parser corruption from
  off-distribution draft tokens (garbled `analysis`/`assistant` headers); SGLang + the LMSYS draft
  produced **none** — higher acceptance means fewer wrong draft tokens reaching the harmony channel.
- **Config:** `--speculative-algorithm EAGLE3 --speculative-draft-model-path lmsys/EAGLE3-gpt-oss-120b-bf16
  --speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`
  (SGLang's documented gpt-oss recipe). Loaded + CUDA-graph captured in **450 s**.
- **Memory 114.08 GB** = SGLang static reservation (default `--mem-fraction-static`) + the 0.9B BF16 draft
  (~1.8 GB) + its KV — a reservation, not the footprint (cf. the base 120b at ~105 GB resident).
- **Takeaway:** EAGLE3 *does* pay off for gpt-oss-120b on the Spark **at single-stream latency** — but
  only with a workload-matched draft. The earlier "gpt-oss EAGLE3 is useless" finding was really
  "the NVIDIA *throughput* draft is useless on general chat." See the conc-32 sibling for whether the win
  survives heavy batching.
