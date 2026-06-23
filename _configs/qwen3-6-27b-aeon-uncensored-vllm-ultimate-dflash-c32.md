---
title: Qwen3.6-27B AEON Uncensored · vLLM-ultimate (custom) · NVFP4 + DFlash · conc 32
model: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored
company: Alibaba
family: Qwen
params: 27B (dense) + DFlash external drafter
engine: vLLM (aeon-vllm-ultimate custom container)
speculative: DFlash (z-lab drafter, num_speculative_tokens 12)
quant: NVFP4 (XS mixed-precision)
quant_rationale: Same AEON NVFP4-XS model served via the card's "DGX Spark production" recipe — custom container ghcr.io/aeon-7/aeon-vllm-ultimate:latest (vLLM 0.23.0) + external z-lab/Qwen3.6-27B-DFlash drafter, DFlash num_speculative_tokens=12. User's explicit request. SAFETY — untrusted image; NO creds, models READ-ONLY.
source_repo: AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
download_url: https://huggingface.co/AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b-aeon-uncensored, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: pending
run_command: |
  # UNTRUSTED third-party container — NO creds, models READ-ONLY. Full recipe on the conc-1 page.
  docker run -d --name aeon-ultimate --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface/hub/models--AEON-7--Qwen3.6-27B-AEON-Ultimate-Uncensored-Multimodal-NVFP4-MTP-XS:/aeon:ro \
    -v ~/models/qwen36-27b-dflash:/drafter:ro \
    --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /aeon/snapshots/4ea4a8d3b8beee13b4e883748bab6221f119cbb0 --served-model-name aeon-ultimate \
    --host 0.0.0.0 --port 8000 --quantization modelopt --mamba-cache-dtype float16 --mamba-block-size 256 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --limit-mm-per-prompt '{"image":4,"video":2}' --mm-encoder-tp-mode data \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 --max-num-batched-tokens 16384 \
    --enable-chunked-prefill --enable-prefix-caching --trust-remote-code \
    --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":12}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model aeon-ultimate \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Custom AEON-ultimate container + DFlash, conc 32.** Part of the 1/8/32 sweep on the card's "DGX Spark
production" recipe (custom vLLM 0.23.0 + external z-lab DFlash drafter). Direct A/B vs the native-MTP-on-stock-vLLM result (**303 tok/s @ conc-32**) — does the custom container + DFlash beat stock + MTP at the same batch? The card claims ~340 tok/s @ c=64 / ~45% DFlash accept. See the
[conc-1 page](qwen3-6-27b-aeon-uncensored-vllm-ultimate-dflash-c1) for the safety posture (untrusted
image, no creds, models read-only) and drafter details.

<!-- results pending -->
