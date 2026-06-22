---
title: Gemma 4 12B · llama.cpp · Q4_K_M + MTP
model: google/gemma-4-12B-it
company: Google
family: Gemma
params: 12B (dense)
engine: llama.cpp
speculative: MTP (Google assistant drafter)
quant: Q4_K_M
quant_rationale: unsloth Q4_K_M base + Google's official MTP drafter (merged GGUF). The only working path for the gemma-4-12B Google drafter — the 12B is gemma4_unified, unservable on stock vLLM/SGLang, but llama.cpp runs it via GGUF and supports --spec-type draft-mtp.
source_repo: unsloth/gemma-4-12b-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-12b-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 32
tags: [gemma-4-12b, Google, Gemma, Q4_K_M, 5-15B, conc-32]
status: done
prefill_toks: 150.26
decode_toks: 202.18
mem_gb: 43.20
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 11:52 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744 (--spec-type draft-mtp). Base + MTP drafter (unsloth).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-12b-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 32 -cb \
    --model-draft /models/MTP/gemma-4-12b-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-12b-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Google's MTP drafter on the 12B — and the spec-decode-vs-concurrency story reaches its limit: barely
any gain.** Google's Gemma-4-12B (unsloth Q4_K_M base) + the official `gemma-4-12B-it-assistant` MTP
drafter, via llama.cpp `--spec-type draft-mtp`.

- **The 12B is `gemma4_unified`** — unservable on the stock vLLM (transformers too old; the bump breaks
  the engine) and on SGLang (no gemma4). **llama.cpp via GGUF is the only engine that runs the 12B at
  all here**, which also makes it the only route to the Google drafter for this model.
- **Workload:** ShareGPT V3, concurrency 32. **750/1000, 18 errors**, **hit the 15-min cap**.
- **Throughput (aggregate, conc 32):** prefill **150.3 tok/s**, decode **202.2 tok/s**. TTFT median
  **2.1 s**, TPOT median **139 ms**.
- **Spec-decode gain shrinks to ~nothing as the base model gets faster:**

  | Model + MTP (llama.cpp, conc 32) | base decode | with MTP | gain |
  |---|---|---|---|
  | Gemma-4-31B | 78.5 | 93.0 | **+18%** |
  | **Gemma-4-12B** | 195.0 | 202.2 | **+4%** |

  Same engine, same drafter mechanism, same concurrency — the only difference is the base model's speed.
  The slow 31B leaves idle GB10 compute between its expensive forward passes for the draft model to use;
  the **2.5× faster 12B already saturates the conc-32 batch**, so there's almost no slack and MTP's draft
  + verify overhead nearly cancels its benefit. Combined with the 31B result, the rule is clear:
  **at high concurrency, speculative decoding helps slow/large models modestly and fast/small models
  barely at all** — the opposite of the dramatic single-stream speedups (unsloth's 3.1×). It remains a
  latency tool, not a throughput tool, under heavy batching.
- **Memory: 43.2 GB** — the Gemma global-attention KV again (≈ the base 12B's 41 GB; the 465 MB draft
  head adds little).
- **Caveat:** conc-32 understates spec-decode; a single-stream latency run would show the full MTP gain.
  Recorded at conc 32 for cross-config comparability.
