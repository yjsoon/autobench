---
title: Gemma 4 E4B · vLLM · FP8 + MTP
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: FP8
quant_rationale: leon-se's FP8-Dynamic (compressed-tensors, the done base run's quant) + Google's official MTP assistant drafter (google/gemma-4-E4B-it-assistant) via vLLM's native gemma-4 MTP path. The spec-decode variant of the done FP8 base — measures the MTP speedup on vLLM at conc 32, the trusted-quant counterpart to the NVFP4+MTP run.
source_repo: leon-se/gemma-4-E4B-it-FP8-Dynamic
download_url: https://huggingface.co/leon-se/gemma-4-E4B-it-FP8-Dynamic
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, FP8, ≤4B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # vllm/vllm-openai:cu130-nightly. leon-se FP8-Dynamic base + Google MTP assistant drafter.
  # vLLM gemma-4 MTP path (--speculative-config method=mtp); assistant shares target KV cache.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly leon-se/gemma-4-E4B-it-FP8-Dynamic \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model leon-se/gemma-4-E4B-it-FP8-Dynamic \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**FP8 base + native vLLM MTP spec-decode for the E4B** — the trusted-quant counterpart to the NVFP4+MTP
run. Same leon-se FP8-Dynamic base as the **done FP8 run** (decode 869.7 tok/s, the fastest decode in the
whole benchmark), now with Google's official MTP assistant drafter (`google/gemma-4-E4B-it-assistant`)
over vLLM's gemma-4 MTP path. Headline question: does MTP add anything at **conc 32** on a model this
small, where the FP8 base already saturates the batch? Spec-decode's win is usually at low concurrency, so
this may come out flat or net-negative — that itself is the datapoint. Record draft acceptance (expect
~0.7 first-position / mean-len ≈3 for `num_speculative_tokens=3`, consistent with the llama.cpp E4B-MTP
run's mean-len 2.88) and the decode delta vs the FP8 base. **Pending run.**
