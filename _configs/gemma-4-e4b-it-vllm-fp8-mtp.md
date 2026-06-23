---
title: Gemma 4 E4B · vLLM · FP8 + MTP
model: google/gemma-4-E4B-it
company: Google
family: Gemma
params: ~4B effective (elastic / MatFormer)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: FP8
quant_rationale: leon-se's FP8-Dynamic (compressed-tensors, the done base run's quant) + Google's official MTP assistant drafter (google/gemma-4-E4B-it-assistant) via vLLM's native gemma-4 MTP path. The spec-decode variant of the done FP8 base — measures the MTP speedup on vLLM at conc 32. Previously BLOCKED on vLLM 0.22; UNBLOCKED on the newer nightly-aarch64 (0.23.1).
source_repo: leon-se/gemma-4-E4B-it-FP8-Dynamic
download_url: https://huggingface.co/leon-se/gemma-4-E4B-it-FP8-Dynamic
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-e4b, Google, Gemma, FP8, ≤4B, conc-32]
status: done
prefill_toks: 1513.8
decode_toks: 1261.48
mem_gb: 107.20
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + Gemma4 MTP head
spec_acceptance: 44% avg draft acceptance · mean acceptance length 2.3 · per-position 0.646/0.418/0.278 (num_speculative_tokens=3)
measured_on: 2026-06-23
completed_at: 2026-06-23 09:19 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # UNBLOCKED on nightly-aarch64 (vLLM 0.23.1rc1) — the newer image resolves both walls that blocked
  # vLLM 0.22 (gemma4_assistant arch recognition AND the TRITON_ATTN heterogeneous-head assertion).
  # No custom image / VLLM_IMAGE override needed — nightly-aarch64 is now the wrapper default.
  scripts/bench-vllm-serving.sh leon-se/gemma-4-E4B-it-FP8-Dynamic 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-E4B-it-assistant","num_speculative_tokens":3}'
  # 1000/1000 prompts, 0 errors, 175.4 s. TTFT median 140.7 ms, TPOT median 21.7 ms.
  # SpecDecoding (steady-state): mean acceptance length ~2.3, avg draft acceptance ~44%,
  # per-position 0.646 / 0.418 / 0.278 (num_speculative_tokens=3).
---

**DONE — UNBLOCKED by the newer vLLM image.** This was blocked on vLLM 0.22 (`autobench-vllm-022-tf`)
by a TRITON_ATTN attention-group assertion; the maintained **`nightly-aarch64` (vLLM 0.23.1rc1)** clears
it and serves the Gemma-4 E4B native-MTP path cleanly. Trusted-quant counterpart to the NVFP4+MTP run
(leon-se FP8-Dynamic base + Google's `google/gemma-4-E4B-it-assistant` MTP drafter).

- **Result (conc 32):** prefill **1513.8** tok/s, decode **1261.48** tok/s aggregate; **1000/1000, 0
  errors** in 175.4 s. Per-stream TTFT median **140.7 ms**, TPOT median **21.7 ms**. Peak mem **107.2 GB**
  (vLLM static KV reservation at util 0.85 + the MTP head).
- **Acceptance: ~44% avg draft acceptance, mean accept-len ~2.3** (per-position 0.646 / 0.418 / 0.278,
  `num_speculative_tokens=3`). Below the ~70%/~3.0 ideal — expected for **ShareGPT general chat** on a
  small E4B drafter (cf. the llama.cpp E4B MTP run's mean-accept-len 2.88 at conc-1). The 4 draft layers
  map onto base layers 22/23 (`gemma4.py` MTP wiring). Acceptance was stable across the run (brief dips
  to ~30–37% in a couple of windows, otherwise ~43–46%).

## History — why it was blocked, and what changed

Originally BLOCKED (2026-06-22) on two walls, both now gone:

1. **Arch not recognized** — `cu130-nightly` (vLLM 0.19.2rc1 / tf 5.6.0) rejected the `gemma4_assistant`
   drafter at config-validation. Needed transformers ≥ ~5.12.
2. **TRITON_ATTN attention-group assertion (the hard blocker on 0.22).** On `autobench-vllm-022-tf`
   (vLLM 0.22.0 + tf 5.12.1) the engine loaded the FP8 weights + `Gemma4MTPModel`, then died in
   KV-cache profiling: `triton_attn.py get_num_attention_heads_from_layers` asserted uniform `num_heads`
   but Gemma-4 E4B has mixed `{8,4}` heads (the heterogeneous-head property that force-pins it to
   TRITON_ATTN). With the MTP draft layer the metadata builder grouped differing head counts → assert.

**`nightly-aarch64` (vLLM 0.23.1rc1) resolves both.** It recognizes `Gemma4MTPModel` out of the box
(no custom-transformers rebuild) AND gets through KV-cache profiling without the `{8,4}` assertion —
the same image still logs *"Gemma4 model has heterogeneous head dimensions … forcing TRITON_ATTN
backend"*, so the backend is still forced, but 0.23.1's metadata builder no longer trips the uniform-
heads assert. **Takeaway:** retry vLLM walls on `nightly-aarch64` before assuming a model-level block.

- **The NVFP4+MTP sibling stays separate** — `nightly-aarch64` regresses Gemma-4 **NVFP4 loading**
  (`gemma4.py tie_weights → NotImplementedError`), so that config can't simply ride this image; see its
  page for the retest.
- **llama.cpp E4B MTP** (`--spec-type draft-mtp`) remains the other working path (conc-1 decode 99.6
  tok/s, mean-accept-len 2.88).
