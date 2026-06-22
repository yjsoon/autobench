---
title: Gemma 4 31B · llama.cpp · Q4_K_M + MTP
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: llama.cpp
speculative: MTP (Google assistant drafter)
quant: Q4_K_M
quant_rationale: unsloth Q4_K_M base + Google's official MTP drafter (merged GGUF) — the only working path to benchmark Google's gemma-4 assistant drafter (SGLang's spark image has no gemma4 support; vLLM rejects gemma multimodal draft-model spec-decode).
source_repo: unsloth/gemma-4-31B-it-GGUF
download_url: https://huggingface.co/unsloth/gemma-4-31B-it-GGUF
context: 65536
modalities: [text]
mm_served: false
concurrency: 32
tags: [gemma-4-31b, Google, Gemma, Q4_K_M, 16-40B, conc-32]
status: done
prefill_toks: 79.37
decode_toks: 92.98
mem_gb: 88.33
mem_source: system MemAvailable delta (10s sampling)
measured_on: 2026-06-22
completed_at: 2026-06-22 11:36 +08
run_command: |
  # ghcr.io/ggml-org/llama.cpp:full-cuda build 9744 (has --spec-type draft-mtp; MTP merged 2026-06-07).
  # Base + Google MTP drafter both under /home/gauravmm/models (unsloth merged GGUFs).
  docker run --gpus all -p 8081:8081 -v /home/gauravmm/models:/models:ro \
    ghcr.io/ggml-org/llama.cpp:full-cuda \
    --server -m /models/gemma-4-31B-it-Q4_K_M.gguf -ngl 99 -c 65536 --parallel 32 -cb \
    --model-draft /models/MTP/gemma-4-31B-it-Q8_0-MTP.gguf --spec-type draft-mtp --spec-draft-n-max 4 -fa on \
    --host 0.0.0.0 --port 8081
  python3 scripts/bench-serving.py --base-url http://localhost:8081 \
    --model gemma-4-31B-it-Q4_K_M.gguf \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Google's official MTP drafter, benchmarked on llama.cpp — the only engine that could run it. It
helps, but far less than vLLM's EAGLE3 at this concurrency.** Google's Gemma-4-31B (unsloth Q4_K_M
base) + the official `gemma-4-31B-it-assistant` MTP drafter (unsloth's merged GGUF), via llama.cpp's
`--spec-type draft-mtp` (MTP merged into llama.cpp 2026-06-07, PR #23398; image build 9744).

- **Why llama.cpp and not the documented engines:** Google's assistant drafter is documented for
  SGLang (`--speculative-algorithm NEXTN`), but `lmsysorg/sglang:spark` has **no gemma4 support**
  (transformers 4.57.1, no `gemma4` model). vLLM **rejects** Gemma draft-model spec-decode (multimodal
  guard, issue #42005). llama.cpp serves gemma4 via GGUF and supports `draft-mtp`, so it's the **only
  working path** for the Google drafter here.
- **Workload:** ShareGPT V3, concurrency 32. **362/1000, 10 errors**, **hit the 15-min cap**.
- **Throughput (aggregate, conc 32):** prefill **79.4 tok/s**, decode **93.0 tok/s**. TTFT median
  **5.6 s**, TPOT median **288 ms**.
- **The MTP gain is real but small at conc 32 — and that's the point:**

  | 31B config | engine | spec | decode | vs its base |
  |---|---|---|---|---|
  | Q4_K_M | llama.cpp | none | 78.5 | — |
  | **Q4_K_M + MTP** | **llama.cpp** | **MTP** | **93.0** | **+18%** |
  | NVFP4 | vLLM | none | 167.0 | — |
  | NVFP4 + EAGLE3 | vLLM | EAGLE3 | 264.7 | +59% |

  unsloth measured this MTP drafter at **3.1×** (52→162 tok/s) — but **single-stream**. At
  **concurrency 32** the batch already saturates the GB10, leaving little idle compute for the draft
  model to exploit and adding verification overhead, so MTP nets only **+18%** here. vLLM's
  **batch-aware** EAGLE3 extracts **+59%** on the same model at the same concurrency. **Speculative
  decoding is both concurrency- and engine-sensitive**: its headline single-stream speedups shrink
  under heavy batching, and a batch-aware implementation (vLLM) beats a simpler one (llama.cpp) when
  the batch is full. For a latency-oriented single-user deployment the MTP number would be far higher;
  for 32-way serving throughput it's a modest gain.
- **Memory: 88.3 GB** — same global-attention KV cliff as the base 31B (the 491 MB draft head is
  negligible); MTP doesn't relieve the footprint, only the per-token latency.
- **Caveat:** conc-32 understates spec-decode. A single-stream (conc 1) latency run would show the
  full MTP benefit; recorded at conc 32 for cross-config comparability.
