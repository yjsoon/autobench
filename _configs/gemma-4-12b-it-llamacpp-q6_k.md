---
title: Gemma 4 12B · llama.cpp · Q6_K
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: llama.cpp
quant: Q6_K
quant_rationale: unsloth Q6_K — the middle rung of the 12B llama.cpp quant ladder. The 12B is gemma4_unified (vLLM/SGLang-blocked), so llama.cpp multi-quant is its only coverage axis.
source_repo: unsloth/gemma-4-12b-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-12b-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, Q6_K, 5-15B, conc-32]
status: done
prefill_toks: 115.67
decode_toks: 159.29
mem_gb: 43.00
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 12:09 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda (dispatcher → --server). GGUF under /home/gauravmm/models.
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-12b-it-Q6_K.gguf -ngl 99 -c 65536 \
    --parallel 32 -cb --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-12b-it-Q6_K.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**The middle rung of the 12B llama.cpp quant ladder.** Google's Gemma-4-12B, unsloth Q6_K.

- **Workload:** ShareGPT V3, concurrency 32. **615/1000, 13 errors**, **hit the 15-min cap**.
- **Throughput (aggregate, conc 32):** prefill **115.7 tok/s**, decode **159.3 tok/s**. TTFT median
  **3.4 s**, TPOT median **166 ms**.
- **The 12B llama.cpp quant ladder — smaller quant decodes faster (bandwidth-bound):**

  | Quant | bits/wt (~) | decode | mem |
  |---|---|---|---|
  | Q4_K_M | 4.5 | 195.0 | 41 GB |
  | **Q6_K** | 6.6 | **159.3** | 43 GB |
  | Q8_0 | 8.5 | 153.0 | 48 GB |

  Decode falls **195 → 159 → 153** as the weights get heavier — the same bandwidth-bound rule the
  vLLM NVFP4/FP8/BF16 ladders showed, here within llama.cpp. (The Q6_K→Q8_0 gap is small; most of the
  cost is already paid going from Q4 to Q6.) For throughput, **Q4_K_M is the sweet spot** on this
  model; the higher quants buy quality at a real decode + memory cost, and all three are dominated by
  Gemma's global-attention KV (41–48 GB).
- **Context:** the 12B is **gemma4_unified**, which the stock vLLM/SGLang can't serve, so this whole
  ladder lives on llama.cpp — the model's only working engine here.
