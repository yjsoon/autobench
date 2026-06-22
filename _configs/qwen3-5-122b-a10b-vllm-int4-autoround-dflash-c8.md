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
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-06-23
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # BLOCKED at KV-cache unification (see Notes) — this is the actual de-risked command attempted,
  # not the original recipe. Substituted nightly-aarch64 (vLLM 0.23.1, rewritten KV unifier) for the
  # recipe's --tf5 build (vLLM 0.22, which hard-asserts on the hybrid page sizes). Pre-downloaded base +
  # DFlash draft to cache; HF_HUB_OFFLINE=1. Dropped recipe extras irrelevant to a throughput bench
  # (custom .jinja, fastsafetensors, flash_attn backend, tool/reasoning parsers, override-gen-config).
  # Added num_speculative_tokens:8 (required by vLLM's dflash) and --enforce-eager (to skip the
  # cudagraph-profiling KV path) — still hit the residual page-size assert.
  docker run -d --name vllm-qwen35-122b --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env HF_TOKEN=*** --env HF_HUB_OFFLINE=1 \
    --env VLLM_ENABLE_CUDAGRAPH_GC=1 --env FLASHINFER_DISABLE_VERSION_CHECK=1 \
    --env VLLM_USE_FLASHINFER_SAMPLER=1 --env VLLM_MARLIN_USE_ATOMIC_ADD=1 \
    vllm/vllm-openai:nightly-aarch64 shieldstar/Qwen3.5-122B-A10B-int4-AutoRound-EC \
    --served-model-name qwen --host 0.0.0.0 --port 8000 \
    --max-model-len 262144 --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 32768 --max-num-seqs 8 \
    --dtype bfloat16 --trust-remote-code --enforce-eager \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.5-122B-A10B-DFlash","revision":"bce6f76cef2027552bed4a8a1bc9c449def48f05","num_speculative_tokens":8}' \
    --enable-prefix-caching --enable-chunked-prefill
  # → RuntimeError: Engine core init failed; AssertionError at
  #   vllm/v1/core/kv_cache_utils.py:1077 (unify_kv_cache_spec_page_size)
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

---

## BLOCKED 2026-06-23 — vLLM cannot unify this hybrid model's KV-cache page sizes. Investigated to the engine source; not flag-fixable here.

**Every preliminary risk flag cleared — the block is a real engine limitation, found only after the model
loads.** Order of investigation:

- **Trust ✓** — `shieldstar` int4-AutoRound-EC base: **42,785 dl/30d**, has `config.json` +
  `quantization_config.json` + chat template in `tokenizer_config`; `z-lab` DFlash drafter: 1.55 GB,
  4502 dl, 17 likes. Both corroborated (same individual-uploader exception as cosmicproc, on the user's
  explicit recipe). Pre-downloaded both to cache (`HF_HUB_OFFLINE=1`, avoids the second-egress cap).
- **DFlash support ✓** — `method: dflash` is **natively in vLLM 0.22+** (`vllm/v1/spec_decode/dflash.py`;
  registered methods include `dflash`). Not a custom plugin. Needs explicit `num_speculative_tokens`
  (set 8, matching z-lab's block-size; the recipe omitted it → first failure, a `pydantic ValidationError`,
  fixed).
- **Truncated fields ✓** — pinned the DFlash `revision` to the repo's current sha
  `bce6f76cef2027552bed4a8a1bc9c449def48f05`; dropped the custom `qwen3.5-enhanced.jinja` (the base repo
  ships a chat template) and the tool/reasoning/override-gen-config flags (irrelevant to a throughput
  bench) — de-risked per ponytail.
- **Weights load ✓** — int4 AutoRound runs on GB10: `MarlinLinearKernel` (linear) + `MARLIN` WNA16 MoE
  backend selected, FLASH_ATTN, 15 shards (~67 GB) loaded. Fit is fine (~74 GB weights + small KV; only
  2 KV heads).

**The hard blocker — KV-cache page-size unification fails for `{GDN linear-attn + full-attn + DFlash draft}`:**
Qwen3.5-122B-A10B is a **hybrid** model — Gated-DeltaNet **linear-attention** layers (`qwen_gdn_linear_attn`,
mamba-style state cache) interleaved with **full-attention** layers — plus the DFlash draft's own KV spec.
vLLM must unify all these into one page size, and can't:
- **vLLM 0.22.0** (the recipe's `--tf5` build, `autobench-vllm-022-tf`): hard
  `assert new_spec.page_size_bytes == max_page_size` in `unify_kv_cache_spec_page_size` → fails immediately.
- **vLLM 0.23.1** (`nightly-aarch64`, the cu130-nightly successor — substituted because its KV unifier was
  rewritten to pad mismatched pages): gets **further** — logs `Setting attention block size to 2192 …`
  + `Padding mamba page size by 0.55% …` — but still trips a **residual**
  `assert page_size_bytes == max_page_size` (`kv_cache_utils.py:1077`) inside `get_kv_cache_groups`. It
  fails first in the cudagraph-memory-profiling path (`profile_cudagraph_memory →
  _init_minimal_kv_cache_for_profiling`); with **`--enforce-eager`** (cudagraph disabled, profiling
  skipped) it then fails at the **same assert in the real path** (`_initialize_kv_caches →
  get_kv_cache_configs`). So `--enforce-eager` does **not** bypass it — the padding doesn't reach exact
  page-size equality for this spec combination.

**Conclusion:** needs an upstream vLLM fix to `unify_kv_cache_spec_page_size` (handle hybrid
linear+full-attention **with** a spec-decode draft KV spec), or a source patch (declined — patching the
sanity assert risks a wrong KV layout / silent corruption, against the trusted-path policy). Same *family*
of "vLLM assertion on heterogeneous cache groups" that blocks Gemma-4 MTP, but a different exact assert
(KV page size vs attention-head grouping). Re-test when vLLM ships a hybrid+spec KV-unification fix.

**Note:** Qwen3.5 is superseded by Qwen3.6 (see the callout above); this was a user-requested standalone
datapoint, not part of the core list. The base model would likely serve **without** DFlash (the draft's
KV spec is a prime suspect for tipping unification over), but a non-spec int4 run is a different config and
wasn't requested — left for a follow-up if wanted.
