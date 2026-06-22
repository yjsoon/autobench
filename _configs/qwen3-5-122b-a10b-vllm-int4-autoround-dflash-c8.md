---
title: Qwen3.5-122B-A10B · vLLM · int4-AutoRound-EC + DFlash · conc 8
model: Qwen/Qwen3.5-122B-A10B
company: Alibaba
family: Qwen
params: 122B / 10B (MoE)
engine: vLLM
speculative: DFlash (z-lab drafter)
quant: int4-AutoRound-EC
quant_rationale: shieldstar's int4 AutoRound-EC (error-corrected AutoRound) base + z-lab's DFlash speculative drafter (z-lab/Qwen3.5-122B-A10B-DFlash), per a user-supplied DGX Spark recipe (builder eugr). Both are INDIVIDUAL-uploader repos (normally a BLOCK per the trusted-repo policy) — added at the user's explicit request. Revives the Qwen3.5 line that was archived as superseded by Qwen3.6; run as a standalone large-MoE spec-decode datapoint.
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
measured_on:
completed_at:
run_command: |
  # User-supplied DGX Spark recipe (builder eugr). Needs the tf5 build = vLLM 0.22 + transformers 5.x
  # (our autobench-vllm-022-tf image, VLLM_IMAGE=). DFlash is z-lab's custom spec method — verify the
  # image's vLLM actually registers method=dflash before trusting the run (see Notes). solo_only.
  # generation_config / speculative_config revision were truncated in the source recipe — fill from the
  # repo before running. A custom chat template (qwen3.5-enhanced.jinja) must be mounted into the
  # container (the recipe applies mods/fix-qwen3.5-enhanced-chat-template).
  docker run -d --name vllm-qwen35-122b --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v "$PWD/qwen3.5-enhanced.jinja:/chat/qwen3.5-enhanced.jinja:ro" \
    --env HF_TOKEN=*** \
    --env VLLM_ENABLE_CUDAGRAPH_GC=1 \
    --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
    --env VLLM_USE_FLASHINFER_SAMPLER=1 \
    --env VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    autobench-vllm-022-tf shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC \
    --served-model-name qwen --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 32768 --max-num-seqs 8 \
    --dtype bfloat16 --load-format fastsafetensors --trust-remote-code \
    --attention-backend flash_attn \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-122B-A10B-DFlash","revision":"6c7242c934a9870d7c59c052..."}' \
    --enable-prefix-caching --enable-chunked-prefill --enable-prompt-tokens-details \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --chat-template /chat/qwen3.5-enhanced.jinja \
    --reasoning-parser qwen3 --generation-config auto \
    --override-generation-config '{"temperature":1.0,"top_p":0.95,"top_k":20,"min_p":0.0,"presence_penalty":1.5,"repetition_penalty":1.0}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model qwen \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 900 --concurrency 8 --max-tokens 256
---

**Pending — user-supplied DGX Spark recipe (builder `eugr`), queued as a standalone Qwen3.5 large-MoE
spec-decode datapoint.** This revives the Qwen3.5-122B-A10B line that was [archived as superseded by
Qwen3.6]; added at the user's explicit request with a specific quant + drafter combo not covered by the
old GPTQ stub.

> **Replaced by → `Qwen/Qwen3.6-35B-A3B`** (sparse-MoE 3.6 counterpart; benchmarked base + native MTP —
> see `qwen3-6-35b-a3b-vllm-fp8`). Qwen's 3.6 line (27B dense + 35B-A3B MoE) outperforms the entire 3.5
> family at a fraction of the size — per [qwen.ai/blog](https://qwen.ai/blog?id=qwen3.6-27b) even the 27B
> dense beats the 397B 3.5 model — which is why this 122B/10B 3.5 MoE was archived. This page exists only
> to capture the user-requested int4-AutoRound-EC + DFlash datapoint; for current Qwen use the 3.6 configs.

- **Stack:** shieldstar `int4-AutoRound-EC` (error-corrected AutoRound 4-bit) base + z-lab **DFlash**
  speculative drafter (`z-lab/Qwen3.5-122B-A10B-DFlash`, `method: dflash`), at **262K context**, 32K
  prefill budget, `--max-num-seqs 8` (conc-8). Qwen general-thinking sampler (`temp 1.0 / top_p 0.95 /
  top_k 20`, `presence_penalty 1.5`), `reasoning-parser qwen3`, `tool-call-parser qwen3_coder`.
- **Image:** the recipe's `--tf5` build (vLLM 0.22 + transformers 5.x) → our **`autobench-vllm-022-tf`**
  (`VLLM_IMAGE=`). Env from the recipe: `VLLM_ENABLE_CUDAGRAPH_GC=1`, `FLASHINFER_DISABLE_VERSION_CHECK=1`,
  `VLLM_USE_FLASHINFER_SAMPLER=1`, `VLLM_MARLIN_USE_ATOMIC_ADD=1`.
- **Recipe metadata (from source):** quant_bits 4 (auto-round, int4), num_kv_heads 2, head_dim 256,
  num_layers 48, `solo_only: true`, `max_nodes: 1`, container `vllm-node-tf5`, mod
  `fix-qwen3.5-enhanced-chat-template`.

**Risk flags to resolve before/at run time** (why this is pending, not yet validated):

1. **Trust:** both `shieldstar` (base quant) and `z-lab` (DFlash drafter) are **individual uploaders** —
   normally a BLOCK per the trusted-repo policy. Running only because the user supplied the full recipe
   explicitly (same exception as the cosmicproc NVFP4 run). Verify both repos look legit (config,
   `hf_quant_config.json`, downloads) at download time.
2. **DFlash method support:** `method: dflash` is **z-lab's custom speculative method**, not a stock vLLM
   one (vLLM ships eagle/eagle3/mtp/ngram/medusa). Confirm the `autobench-vllm-022-tf` vLLM actually
   registers `dflash` (it may need a z-lab plugin / `--speculative-config` extension) before trusting the
   run — if vLLM rejects the method, this is a BLOCK like the Gemma-4 MTP configs.
3. **Truncated fields:** the source recipe truncated the `generation_config` (repetition_penalty + tail)
   and the DFlash `revision` pin (`6c7242c9…`). Fill both from the repos before running; the command above
   carries placeholders.
4. **Custom chat template:** `qwen3.5-enhanced.jinja` (+ the `fix-qwen3.5-enhanced-chat-template` mod) is a
   recipe artifact not in this repo — obtain and mount it, or the chat endpoint won't format correctly.
5. **Fit:** int4 122B-A10B ≈ ~60 GB weights + 262K-ctx KV at util 0.85 on 121 GB unified — should fit but
   watch the MemAvailable delta; drop `--max-model-len` if KV profiling OOMs.

**Capture at run time:** DFlash acceptance length / per-position acceptance (the wrapper greps
`accept|spec.?decode` before teardown), decode speedup vs a non-spec int4 base, and the TRT-LLM-style
engine/build cost if any. Cross-check DFlash acceptance against z-lab's published number and flag any gap.
