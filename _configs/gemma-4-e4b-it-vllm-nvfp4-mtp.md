---
title: Gemma 4 E4B · vLLM · NVFP4 + MTP
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: cosmicproc NVFP4 (W4A4, NVIDIA ModelOpt) base + Google's official MTP assistant drafter (google/gemma-4-E4B-it-assistant) via vLLM's native gemma-4 MTP path (--speculative-config method=mtp). The NVFP4 base on Blackwell + spec-decode stacked. Individual-uploader base quant added at user request; cross-check acceptance vs the FP8+MTP variant to flag any quant/drafter mismatch.
source_repo: cosmicproc/gemma-4-E4B-it-NVFP4
download_url: https://huggingface.co/cosmicproc/gemma-4-E4B-it-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, NVFP4, ≤4B, conc-32]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # vllm/vllm-openai:cu130-nightly. cosmicproc NVFP4 base + Google MTP assistant drafter.
  # vLLM gemma-4 MTP path: --speculative-config method=mtp (NOT a separate draft_model). The
  # assistant shares the target KV cache; needs a vLLM build with gemma-4 MTP support (logs must
  # show SpeculativeConfig(method='mtp'), not method='draft_model').
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly cosmicproc/gemma-4-E4B-it-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model cosmicproc/gemma-4-E4B-it-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**NVFP4 base + native vLLM MTP spec-decode for the E4B.** Stacks cosmicproc's NVFP4 (W4A4) base with
Google's official MTP assistant drafter (`google/gemma-4-E4B-it-assistant`) over vLLM's gemma-4 MTP
path (`--speculative-config method=mtp` — the assistant shares the target's KV cache; **not** the
separate-draft-model path). Compare decode tok/s against the NVFP4 base (no MTP) to size the spec
speedup at conc 32, and **cross-check draft acceptance against the FP8+MTP run** — a large gap would flag
an NVFP4-base / BF16-drafter mismatch. **Pending run.**

> The old CLAUDE.md note "vLLM rejects gemma spec-decode" is **stale** — recent vLLM ships native
> gemma-4 MTP (E2B/E4B/12B/26B-A4B/31B assistants). Two risks remain for *this* config: (1) the
> cu130-nightly build must actually include the MTP path (else it logs `method='draft_model'` and falls
> back); (2) MTP layered on an NVFP4 (W4A4) base is unproven — if EngineCore rejects it, this drops to
> `status: blocked` and only the FP8+MTP variant carries the spec datapoint.
