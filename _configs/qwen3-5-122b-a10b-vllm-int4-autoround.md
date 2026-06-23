---
title: Qwen3.5-122B-A10B · vLLM · int4-AutoRound-EC · conc 8
model: Qwen/Qwen3.5-122B-A10B
company: Alibaba
family: Qwen
params: 122B / 10B (MoE)
engine: vLLM
quant: int4-AutoRound-EC
quant_rationale: shieldstar's int4 AutoRound-EC (error-corrected AutoRound) base, served WITHOUT the DFlash drafter — the non-speculative counterpart to qwen3-5-122b-a10b-vllm-int4-autoround-dflash-c8. The DFlash config is BLOCKED (vLLM can't unify the hybrid GDN+full-attn KV page sizes WITH a draft KV spec); the Notes there suspected the draft KV spec is what tips unification over, so a base-only run is the test of that hypothesis. Individual-uploader repo (normally a BLOCK), added at the user's explicit request alongside the DFlash recipe.
source_repo: shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC
download_url: https://huggingface.co/shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC
context: 262144
modalities: [text]
mm_served: false
concurrency: 8
tags: [qwen3.5-122b-a10b, Alibaba, Qwen, int4-AutoRound-EC, 41-130B, conc-8]
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). Same base as the DFlash config, with the --speculative-config DROPPED.
  # Tests whether the hybrid GDN+full-attn KV unifies once the draft's KV spec is removed (the
  # blocked DFlash page suspected the draft spec is what tips unification over).
  docker run -d --name vllm-qwen35-122b --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN=*** --env HF_HUB_OFFLINE=1 \
    vllm/vllm-openai:nightly-aarch64 shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC \
    --served-model-name qwen --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 32768 --max-num-seqs 8 \
    --dtype bfloat16 --trust-remote-code --enable-prefix-caching --enable-chunked-prefill
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model qwen \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 900 --concurrency 8 --max-tokens 256
---

**Queued — Qwen3.5-122B-A10B int4-AutoRound-EC WITHOUT DFlash.** The non-speculative counterpart to the
[blocked DFlash config](qwen3-5-122b-a10b-vllm-int4-autoround-dflash-c8); requested as the follow-up that
page explicitly left open.

> **Why this might serve where DFlash didn't:** the DFlash run BLOCKED at vLLM's KV-cache page-size
> unification for `{GDN linear-attn + full-attn + DFlash draft}`. Its Notes flagged the **draft's KV spec
> as the prime suspect** for tipping unification over. Dropping `--speculative-config` removes that spec —
> if the model loads here, it confirms the draft was the trigger; if it still asserts at
> `unify_kv_cache_spec_page_size`, the hybrid GDN+full-attn cache alone is the wall (a separate finding
> worth recording, and grounds to BLOCK this too).

- **Same** shieldstar int4-AutoRound-EC base, 262K ctx, conc-8, `nightly-aarch64` (vLLM 0.23.1 rewritten
  KV unifier). Individual-uploader repo, run per the user's explicit request (same exception as the
  DFlash sibling / cosmicproc). Pre-download to cache, `HF_HUB_OFFLINE=1`.
- **Qwen3.5 is superseded by Qwen3.6** (see the DFlash page's callout) — this is a standalone
  large-MoE datapoint, not part of the core list.
