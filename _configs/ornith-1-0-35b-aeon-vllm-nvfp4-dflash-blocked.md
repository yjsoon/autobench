---
title: Ornith-1.0-35B AEON Uncensored · vLLM (aeon-ultimate) · NVFP4 + DFlash · conc 1 — BLOCKED
model: AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
company: Alibaba
family: Qwen
params: Qwen3.6-35B-A3B base (35B/3B MoE, hybrid GDN+full-attn) · AEON "Ornith 1.0" RL post-train, abliterated
engine: vLLM (aeon-vllm-ultimate custom container, v0.23.0+aeon.sm121a.dflash)
speculative: DFlash (z-lab/Qwen3.6-35B-A3B-DFlash, num_speculative_tokens 6) — BLOCKED at KV unification
quant: NVFP4 (compressed-tensors, nvfp4-pack-quantized)
quant_rationale: AEON's headline config — NVFP4 base + DFlash drafter for a claimed ~1.9× single-stream speedup. The base runs fine (see the conc-32 / conc-1 pages); adding the DFlash draft is what blocks.
source_repo: AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
download_url: https://huggingface.co/AEON-7/Ornith-1.0-35B-AEON-Ultimate-Uncensored-NVFP4
context: 262144
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [ornith-1-0-35b-aeon, Alibaba, Qwen, NVFP4, 16-40B, conc-1]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-06-28
completed_at: 2026-06-28 16:13 +0800
engine_image: ghcr.io/aeon-7/aeon-vllm-ultimate:latest@sha256:be9e05a11da6e72607ab6f3e960993b253b673af0727005122a3266129a518e3
run_command: |
  # BLOCKED — boots through drafter load, then hard-asserts at KV-cache page-size unification.
  # Untrusted image — NO creds, model + drafter READ-ONLY, loopback port.
  docker run -d --name serving-ornith --gpus all --ipc=host \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e CUTE_DSL_ARCH=sm_121a -e VLLM_USE_FLASHINFER_SAMPLER=1 \
    -v ~/ornith/model:/model:ro -v ~/ornith/drafter:/drafter:ro -p 127.0.0.1:8000:8000 \
    --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:latest \
    serve /model --served-model-name ornith --host 0.0.0.0 --port 8000 \
    --quantization compressed-tensors --trust-remote-code \
    --max-model-len 262144 --gpu-memory-utilization 0.6 \
    --max-num-seqs 16 --max-num-batched-tokens 16384 \
    --mamba-cache-dtype float32 --enable-prefix-caching --reasoning-parser qwen3 \
    --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":6}'
  # LIKELY UNBLOCK (untested): pin the drafter to its pre-"Modal retrain" revision (small-page arch):
  #   --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.6-35B-A3B-DFlash",
  #     "revision":"31977fbe13a8...","num_speculative_tokens":6}'
---

**BLOCKED — DFlash + this hybrid MoE trips the same KV-cache page-size unification assert that blocked
the [Qwen3.5-122B-A10B DFlash run](qwen3-5-122b-a10b-vllm-int4-autoround-dflash-c8). Same root cause,
same likely fix (pin the draft to a small-page revision).**

**Boot gets all the way through drafter load, then asserts during KV setup:**
- `Resolved architecture: DFlashDraftModel`; `speculative_config=SpeculativeConfig(method='dflash',
  num_spec_tokens=6)`; drafter loaded (EAGLE-style: shares target embed/lm_head, aux layers
  `(2,7,12,17,23,28,33,38)`).
- Then, in `determine_available_memory → profile_cudagraph_memory →
  _init_minimal_kv_cache_for_profiling → get_kv_cache_groups → unify_kv_cache_spec_page_size`:
  ```
  vllm/v1/core/kv_cache_utils.py:1064:  assert new_spec.page_size_bytes == max_page_size
  AssertionError  →  EngineCore init failed (exit 1)
  ```

**Why:** Ornith is a **hybrid** model (Gated-DeltaNet linear-attention + full-attention, plus a mamba/SSM
state cache). Without a draft, vLLM unifies the two KV specs by padding the attention page to the mamba
page (the base conc-32/conc-1 runs log `Setting attention block size to 1072 …` and serve fine). Adding
the **DFlash draft introduces a third KV spec**; on vLLM 0.23 (the AEON fork's base) it cannot reconcile
all three into one page size. This is the **identical failure family** documented in
`notes/INCOMPATIBILITIES.md` and confirmed by the base-vs-draft A/B here (base serves, +draft asserts) —
it is specifically the **draft's KV spec**, not the hybrid base.

**The draft arch is the large-page variant — and that's the unblock lever.** The current
`z-lab/Qwen3.6-35B-A3B-DFlash` `main` config is `sliding_window: 4096`, `num_key_value_heads: 8`,
`num_hidden_layers: 6` — the *same* big-page shape (sw 4096 / 8 kv-heads) that broke the 122B on its
`bce6f76` rev. The 122B was **unblocked** by pinning the draft to an older small-page rev (`6c7242c`:
sw 2048 / 4 kv-heads), which pads to an equal page size. For this drafter the **pre-"Modal retrain"
revisions** (`31977fbe13a8` "Upload model", `f98dc5c2908b`) have `num_key_value_heads: 4` and **no
sliding window** — the analogous small-page arch. **Untested hypothesis (high confidence):** pinning the
DFlash `revision` to `31977fbe13a8` lets the page sizes unify and DFlash boots, exactly as the 122B fix.
(The retrained `main` weights would not be used — a tradeoff to validate.)

**Bottom line:** AEON's headline "~1.9× DFlash" does **not** reproduce on this hybrid model with the
shipped (retrained) drafter — it asserts at boot. The base model is fast and stable without it (decode
422 tok/s conc-32, 37.7 tok/s conc-1 @ 256K). Next step is to test the pinned-small-page-draft unblock;
until then this config is **blocked**, and the trusted recommendation is native MTP or DFlash-off.

> Cross-ref: [`qwen3-5-122b-a10b-vllm-int4-autoround-dflash-c8`](qwen3-5-122b-a10b-vllm-int4-autoround-dflash-c8)
> (same assert, unblocked by a draft-revision pin) and `notes/INCOMPATIBILITIES.md` (hybrid+spec KV
> unification wall).
