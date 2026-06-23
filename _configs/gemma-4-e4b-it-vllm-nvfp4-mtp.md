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
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-06-23
completed_at:
run_command: |
  # STILL BLOCKED after the 2026-06-23 retest — now by MUTUALLY-EXCLUSIVE image requirements (see Notes):
  #  • nightly-aarch64 (0.23.1) fixes the TRITON_ATTN+MTP assertion (FP8+MTP works there) BUT regresses
  #    NVFP4 *loading*: lm_head.tie_weights → quant_method.tie_weights → NotImplementedError.
  #  • cu130-nightly LOADS NVFP4 fine (the done NVFP4 base used it) but is too old for the gemma4_assistant
  #    MTP drafter arch. No single image does both.
  # Retest command (fails at weight-load on nightly-aarch64):
  scripts/bench-vllm-serving.sh cosmicproc/gemma-4-E4B-it-NVFP4 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":3}'
---

## STILL BLOCKED after 2026-06-23 retest — now by mutually-exclusive image requirements

**The FP8+MTP sibling was UNBLOCKED on `nightly-aarch64` (vLLM 0.23.1) — but NVFP4+MTP can't follow it.**
Retesting on the newer image surfaced a *different* wall: `nightly-aarch64` fixes the TRITON_ATTN+MTP
assertion (so FP8+MTP serves), **but it regresses Gemma-4 NVFP4 loading** — EngineCore dies at weight
load with `lm_head.tie_weights(embed_tokens)` → `quant_method.tie_weights` → **`NotImplementedError`**
(`vocab_parallel_embedding.py:560` / `base_config.py:55`), before MTP is even exercised. Meanwhile the
*older* `cu130-nightly` loads NVFP4 fine (the [NVFP4 base, no-MTP] config runs clean there — decode
1073.8 tok/s, the benchmark's fastest) but is too old for the `gemma4_assistant` MTP drafter arch.
**So the two requirements are mutually exclusive on the images available today** — no single vLLM build
both loads Gemma-4 NVFP4 *and* runs the MTP drafter. Revisit when a 0.23+ image fixes NVFP4 `tie_weights`
(then this should work exactly like the FP8+MTP run). **Working E4B MTP path meanwhile = llama.cpp**
(`--spec-type draft-mtp`, conc-1 mean-accept-len 2.88).

---

### Original block (2026-06-22, on vLLM 0.22) — the heterogeneous-head + MTP attention bug
Intended to stack cosmicproc's NVFP4 (W4A4) base with Google's `google/gemma-4-E4B-it-assistant` MTP
drafter over vLLM's gemma-4 MTP path. The blocker was **not** the NVFP4 base (it runs clean on
cu130-nightly — decode 1073.8 tok/s); it was vLLM 0.22's MTP path on Gemma-4's heterogeneous-head
attention:

- On the arch-aware image (`scripts/Dockerfile.vllm-022-tf` = vLLM 0.22 + transformers 5.12.1) vLLM
  resolves `Gemma4MTPModel` + `SpeculativeConfig(method='mtp')`, then **EngineCore dies in KV-cache
  profiling**: `triton_attn.py → get_num_attention_heads_from_layers` →
  `AssertionError: All layers in one attention group must share num_heads; got {8, 4}`.
- Gemma-4 E4B's mixed `num_heads` (8/4) force vLLM onto `TRITON_ATTN`, whose MTP-enabled metadata builder
  can't group the differing head counts. The `{8,4}` heads are language-model layers, so this is
  **independent of base quant** — identical to the FP8+MTP failure. Not flag-fixable.
- See the [FP8+MTP config] for the full traceback and reproduction. **Working E4B MTP path = llama.cpp**
  (already measured, conc-1, mean-accept-len 2.88). Revisit when vLLM supports gemma-4 MTP under its
  heterogeneous-head attention backend.

> Note: the old CLAUDE.md "vLLM rejects gemma spec-decode" is **partly** stale — vLLM *does* now ship
> native gemma-4 MTP (it resolves the arch and config), but it's **not yet usable for the E4B on GB10**
> because of the TRITON_ATTN grouping assertion above.
