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
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-06-22
completed_at:
run_command: |
  # BLOCKED — see Notes. Requires a vLLM build new enough for the gemma4_assistant MTP arch
  # (transformers >= ~5.12), which the stock cu130-nightly (vLLM 0.19.2rc1 / transformers 5.6.0)
  # lacks. Built scripts/Dockerfile.vllm-022-tf (vLLM 0.22 + transformers 5.12.1) to get past the
  # arch-recognition error — but it then hits a TRITON_ATTN attention-group assertion (see Notes).
  VLLM_IMAGE=autobench-vllm-022-tf scripts/bench-vllm-serving.sh \
    leon-se/gemma-4-E4B-it-FP8-Dynamic 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":3}'
---

**BLOCKED — vLLM's gemma-4 MTP path can't build attention metadata for the heterogeneous-head E4B
(reproducible engine crash, two distinct images).** Intended as the trusted-quant counterpart to the
NVFP4+MTP run (leon-se FP8-Dynamic base + Google's `google/gemma-4-E4B-it-assistant` MTP drafter). Two
blockers, in order:

1. **Arch not recognized (stock image).** `vllm/vllm-openai:cu130-nightly` (vLLM **0.19.2rc1**,
   transformers **5.6.0**) rejects the drafter at config-validation: *"checkpoint has model type
   `gemma4_assistant` but Transformers does not recognize this architecture."* Fixed by building
   **`scripts/Dockerfile.vllm-022-tf`** = vLLM **0.22.0** + `pip install -U transformers` (→ **5.12.1**,
   which resolves `gemma4_assistant`).

2. **TRITON_ATTN attention-group assertion (the hard blocker).** On the fixed image vLLM *does* resolve
   `Gemma4MTPModel` + `SpeculativeConfig(method='mtp')` and loads the FP8 weights, then **EngineCore dies
   during KV-cache profiling**:

   ```
   vllm/v1/attention/backends/triton_attn.py → get_num_attention_heads_from_layers
   AssertionError: All layers in one attention group must share num_heads;
   got {8, 4} for ['language_model.model.layers.4.self_attn.attn', ...]
   ```

   Gemma-4 E4B has **heterogeneous attention** (mixed `num_heads` 8/4, the same arch property behind its
   256/512 head dims). vLLM is **force-pinned to the `TRITON_ATTN` backend** *because* of those
   heterogeneous head dims — but with MTP enabled the draft layer makes TRITON_ATTN's metadata builder
   group layers with differing head counts, tripping a uniform-`num_heads` assertion. Not flag-fixable:
   the backend can't be switched (it's forced), and `num_speculative_tokens=1` / `enforce_eager` don't
   change the grouping. Matches the known vLLM issue *"Gemma 4 assistant speculative decoding does not
   match actual behavior"*.

- **Reproduced twice** (wrapper run + a manual direct-launch debug container) — not a transient spawn
  deadlock (those recover on retry; this fails identically every time at the same assertion).
- **Same root cause blocks the NVFP4+MTP config** — the `{8,4}` heads come from the language-model layers,
  independent of base quant. So both vLLM E4B MTP configs are blocked on this vLLM limitation.
- **The working E4B MTP path is llama.cpp** (`--spec-type draft-mtp`), already measured at conc-1
  (decode 99.6 tok/s, mean-accept-len 2.88). Revisit vLLM MTP when a build lands that supports MTP under a
  heterogeneous-head attention backend (or relaxes the TRITON_ATTN grouping assertion).
