---
title: MiniMax-M2.7 · llama.cpp · UD-Q3_K_XL
model: MiniMaxAI/MiniMax-M2.7
company: MiniMax
family: MiniMax-M2
params: ~230B (≈10B active, MoE)
engine: llama.cpp
quant: UD-Q3_K_XL
quant_rationale: "Next smaller trusted Unsloth dynamic quant below the completed `UD-IQ4_XS` run. Verified from `unsloth/MiniMax-M2.7-GGUF` on 2026-06-23: `UD-Q3_K_XL` is about 101.9 GB vs `UD-IQ4_XS` at 108.4 GB, buying ~6.5 GB more headroom on the 121 GB GB10. Queue this next to see whether a slightly smaller runnable quant materially improves the pathological conc-32 result without dropping all the way to the tiny IQ2/IQ3 tiers."
source_repo: unsloth/MiniMax-M2.7-GGUF
download_url: https://huggingface.co/unsloth/MiniMax-M2.7-GGUF/tree/main/UD-Q3_K_XL
context: 131072
modalities: [text]
concurrency: 32
tags: [minimax-m2-7, MiniMax, MiniMax-M2, Q3_K_XL, 130B+, conc-32]
status: pending
run_command: |
  # llama-server (NGC dispatcher image) + ShareGPT serving benchmark, conc=32
  docker run --rm --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/minimax-m2-7/UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf \
    -ngl 99 -c 131072 --parallel 32 -cb \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model minimax-m2-7/UD-Q3_K_XL/MiniMax-M2.7-UD-Q3_K_XL-00001-of-00004.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Queued — the next smaller runnable MiniMax-M2.7 GGUF after `UD-IQ4_XS`.**

- **Why this quant next:** `UD-IQ4_XS` already proved the largest runnable ~4-bit MiniMax path is too
  close to the memory ceiling at **117.78 GB** measured and far too slow at **conc-32**. The next
  smaller trusted Unsloth option is `UD-Q3_K_XL`, which local notes and the Hugging Face repo both place
  below it in size.
- **Comparison setup:** keep **the same engine and concurrency** as the done IQ4_XS run, but raise the
  context target to **128K (`131072`)** per the latest queue direction. This stops being a pure
  apples-to-apples quant comparison and instead tests whether the smaller Q3 quant can buy back a much
  larger usable window on Spark.
- **What this should answer:** whether ~6.5 GB of recovered headroom is enough to support a materially
  larger **128K** server context on Spark while still serving at conc-32, and whether that reduces the
  likely per-slot-context failures seen on the `UD-IQ4_XS` run.
- **What it probably will not answer:** if HTTP 400s persist here too, that likely means the dominant
  issue is still the `32768 / 32 slots` serving shape rather than this specific quant. But it is still
  the right next rung on the trusted-quant ladder.
