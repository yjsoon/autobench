---
title: Qwen3.6-27B AEON Uncensored · vLLM-ultimate (custom) · NVFP4 + DFlash · conc 1
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + DFlash external drafter
engine: vLLM (aeon-vllm-ultimate custom container)
speculative: DFlash (z-lab drafter, num_speculative_tokens 12)
quant: NVFP4 (XS mixed-precision)
quant_rationale: Same AEON NVFP4-XS model as the native-MTP configs, but served via the card's "DGX Spark production" recipe — the custom third-party container ghcr.io/aeon-7/aeon-vllm-ultimate:latest (vLLM 0.23.0) + the external z-lab/Qwen3.6-27B-DFlash drafter mounted at /drafter, DFlash spec-decode num_speculative_tokens=12. Run at the user's explicit request, reversing the earlier "untrusted container declined" call. SAFETY — image is untrusted; run with NO credentials (HF_TOKEN withheld), both models mounted READ-ONLY.
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-1]
status: pending
run_command: |
  # UNTRUSTED third-party container — run with NO creds, models READ-ONLY (model already cached).
  docker run -d --name aeon-ultimate --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface/hub/models--AEON-7--Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS:/aeon:ro \
    -v ~/models/qwen36-27b-dflash:/drafter:ro \
    --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /aeon/snapshots/4ea4a8d3b8beee13b4e883748bab6221f119cbb0 --served-model-name aeon-ultimate \
    --host 0.0.0.0 --port 8000 --quantization modelopt --mamba-cache-dtype float16 --mamba-block-size 256 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --limit-mm-per-prompt '{"image":4,"video":2}' --mm-encoder-tp-mode data \
    --gpu-memory-utilization 0.85 --max-num-seqs 1 --max-num-batched-tokens 16384 \
    --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
    --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":12}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model aeon-ultimate \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 1 --max-tokens 256
---

**The custom AEON-ultimate container + DFlash path, single-stream.** Reverses the earlier decision to
decline the untrusted container (per the user's explicit request). Compares the card's "DGX Spark
production" recipe (custom vLLM 0.23.0 + external z-lab DFlash drafter, `num_speculative_tokens=12`)
against the native-MTP-on-stock-vLLM result. conc-1 is the single-stream latency point of the sweep.

- **Safety posture:** `ghcr.io/aeon-7/aeon-vllm-ultimate:latest` is an individual-org image — run with
  **HF_TOKEN withheld**, **no `.env`/creds** passed, both the AEON model and the DFlash drafter mounted
  **read-only**. The model is already cached, so the container needs no secrets and no write access.
- **Drafter:** `z-lab/Qwen3.6-27B-DFlash` (3.3 GB, 68k dl/mo — the DFlash authors) at `/drafter`.

<!-- results pending — runs after the gpt-oss sweep + DiffusionGemma + MiniMax free the GPU -->
