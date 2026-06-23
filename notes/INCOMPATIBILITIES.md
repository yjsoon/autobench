# Incompatibilities & gotchas

Engine/model-specific failures and their fixes, learned the hard way on GB10/Spark.
Referenced from `CLAUDE.md`. When a new wall is hit, record it here.

## llama.cpp

- **Dispatcher entrypoint:** `ghcr.io/ggml-org/llama.cpp:full-cuda` (ARM64) uses a dispatcher
  entrypoint — run llama-bench via `--bench`, the server via `--server` (not the bare binary names).
- **gpt-oss = blocked (harmony):** llama-server's OpenAI chat endpoint can't parse gpt-oss's
  **harmony** channel output on build `b9744` — every `/v1/chat/completions` request 500s with
  "does not match the expected peg-native format" (raw `/completion` works, `--jinja` /
  `--reasoning-format none` don't help). Benchmark gpt-oss on **vLLM/SGLang** instead. Other
  (non-harmony) models serve fine on llama.cpp.

## vLLM

- **Transient EngineCore spawn deadlock on first launch — kill & retry.** Seen twice: a fresh
  `vllm serve` hangs at init (log frozen at `Enabled custom fusions: act_quant`, **no** EngineCore
  startup line, container ~0.2% CPU / weights never loaded). It's a **spawn race, not an
  incompatibility** — a clean `docker rm -f` + relaunch loads fine. Distinguish from a *real* failure
  (which prints a Python traceback / non-zero exit and reproduces every time): the deadlock is silent
  and clears on retry. Don't mark a config blocked on the first hang.
- **Nemotron NVFP4 headliners:** vLLM is the documented path
  (`--trust-remote-code --reasoning-parser nemotron_v3`); `cu130-nightly` serves
  Nemotron-3-Super-120B-A12B-NVFP4 fine. An *older* NGC vLLM container rejected its
  `quant_algo: MIXED_PRECISION` against a hardcoded whitelist — use **cu130-nightly, not NGC**, for the
  NVFP4 Nemotrons.

### gpt-oss + vLLM — harmony tiktoken-vocab egress gotcha
At startup vLLM's `OpenAIServingResponses` loads the **harmony** encoding, whose rust `openai_harmony`
lib downloads `o200k_base.tiktoken` (sha256 `446a9538…`) from `$TIKTOKEN_ENCODINGS_BASE` (default =
openaipublic blob). The HF *model* download already used the run's **one** allowed external connection,
so this **second** fetch is egress-capped → `HarmonyError: failed to download or load vocab file` and
the server never comes up. Fix (baked into `scripts/bench-vllm-serving.sh`): pre-seed the vocab at
`~/models/tiktoken_cache/o200k_base.tiktoken`, mount it `-v …:/vocab:ro`, and set
**`TIKTOKEN_ENCODINGS_BASE=/vocab`** (a **plain path**, NOT `file:///vocab` — the rust file:// parser
fails with "No such file or directory"). No second egress then needed. Same applies to any
harmony/gpt-oss model on SGLang.

### Mistral native-format + MLA (Mistral-Small-4-119B-NVFP4)
The official NVFP4 repo (`mistralai/Mistral-Small-4-119B-2603-NVFP4`) ships **Mistral native format**
(`params.json`, `consolidated-*.safetensors`, `tekken.json` — no `config.json`), so vLLM needs
`--tokenizer-mode mistral --config-format mistral --load-format mistral`. The model is an **MLA**
(DeepSeek-V2-style) + Pixtral-vision arch, and vLLM's **Triton MLA decode kernel fails to compile on
GB10** (`triton_decode_attention` → `Cannot make_shape_compatible: incompatible dimensions 256 and
512`) — every request kills EngineCore. Fix: **`VLLM_MLA_DISABLE=1`** (use standard attention,
bypassing the broken kernel) **+ `--gpu-memory-utilization 0.90`** (MLA-disabled materializes full KV,
so 0.85 under-sizes the pool for 65536 ctx). `FLASHINFER_MLA` override is *ignored* (falls back to
Triton MLA) — disabling MLA is the only path that works on this build.

### Gemma-4 MTP on vLLM — was BLOCKED on 0.22, FIXED on nightly-aarch64 (0.23.1)
vLLM ships native gemma-4 MTP
(`--speculative-config '{"method":"mtp","model":"google/gemma-4-E<size>-it-assistant","num_speculative_tokens":N}'`).
Two walls historically blocked it on the older images; **both are resolved on `nightly-aarch64` (vLLM
0.23.1rc1)** — FP8+MTP now serves cleanly (measured 2026-06-23: E4B conc-32 decode ~1261 tok/s, accept
~44% / mean-len ~2.3; see `gemma-4-e4b-it-vllm-fp8-mtp`). **Retry vLLM walls on `nightly-aarch64` before
declaring a model-level block.** The history, for reference:
- **Wall 1 — arch not recognized:** the drafter's `gemma4_assistant` arch needs **transformers ≥ ~5.12**;
  stock `cu130-nightly` (vLLM 0.19.2rc1 / tf 5.6.0) rejected it at config-validation. (Worked around on
  0.22 by building `scripts/Dockerfile.vllm-022-tf` = vLLM 0.22.0 + tf 5.12.1.) `nightly-aarch64` ships a
  new-enough transformers and recognizes `Gemma4MTPModel` out of the box.
- **Wall 2 — TRITON_ATTN attention-group assertion (the hard blocker on 0.22):** `triton_attn.py
  get_num_attention_heads_from_layers` asserted uniform `num_heads` per group, but Gemma-4 E4B has
  **mixed `{8,4}` heads** (head_dim 256/512) which *force-pin* it to TRITON_ATTN; with MTP's draft layer
  the 0.22 metadata builder grouped differing head counts → `AssertionError`. **0.23.1 still forces
  TRITON_ATTN** (logs the heterogeneous-head message) but its metadata builder no longer trips the assert.
- **Caveat — NVFP4+MTP not yet on this image:** `nightly-aarch64` **regresses Gemma-4 NVFP4 *loading***
  (`gemma4.py tie_weights → NotImplementedError`, see image policy), so the NVFP4+MTP config can't simply
  ride it — needs retest when NVFP4 loading is fixed on a 0.23+ image. **llama.cpp** (`--spec-type
  draft-mtp`, unsloth merged-GGUF drafter; E-series needs `-fa off`) remains a working path for both.

### Hybrid GDN+full-attn KV unifies on its own, but NOT with a third (spec-decode draft) KV spec
Qwen3.5-122B-A10B is a **hybrid** model — Gated-DeltaNet **linear-attention** (mamba-style state cache)
interleaved with **full-attention** layers. vLLM must unify these into one KV page size.
- **Base (no spec-decode) serves fine** on `nightly-aarch64` (vLLM 0.23.1): the GDN + full-attn cache
  unifies (the 0.23 unifier pads mismatched pages), loads, and benchmarks cleanly — measured 2026-06-23,
  decode 85.5 tok/s conc-8, 0 errors (`qwen3-5-122b-a10b-vllm-int4-autoround`).
- **Adding a DFlash draft → BLOCKED.** The same model **+ a `--speculative-config` DFlash drafter** adds a
  *third* KV spec and trips `assert page_size_bytes == max_page_size` in
  `unify_kv_cache_spec_page_size` / `kv_cache_utils.py:1077` (fails in both the cudagraph-profiling and
  real paths; `--enforce-eager` does not bypass it). **CONFIRMED by the base-vs-draft A/B:** it is
  specifically the draft's KV spec vLLM can't reconcile with the two-way hybrid, **not** the hybrid base.
  Needs an upstream hybrid+spec KV-unification fix. Full trace in
  `qwen3-5-122b-a10b-vllm-int4-autoround-dflash-c8`. (Same *family* as the Gemma-4-MTP-on-0.22 assert, but
  a different exact assert — KV page size vs attention-head grouping.) int4 AutoRound itself runs fine on
  GB10 via AutoGPTQ `MarlinLinearKernel` + `int_wna16` MARLIN MoE.

## SGLang

- **`lmsysorg/sglang:spark` is too old for the Qwen3.6 (`qwen3_5`) arch — FIXED by a newer nightly.**
  The `spark` image ships **transformers 4.57.1**; loading any Qwen3.6 checkpoint fails at config load
  with `ValueError: model type 'qwen3_5' but Transformers does not recognize this architecture`.
  **Fix (measured 2026-06-23): use `lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed`** (arm64 manifest
  exists; transformers **5.8.1**, ships `qwen3_moe`/`qwen3_vl*` model classes). It loads Qwen3.6-27B-NVFP4
  cleanly (`type=Qwen3_5ForConditionalGeneration`, `quant=compressed-tensors`, weights 24.55 GB) and
  benchmarks fine (27B NVFP4 base decode ~178 tok/s, vs vLLM 188). Pass it via the new **`SGLANG_IMAGE=`**
  override on `scripts/bench-sglang-serving.sh`. Benign load warnings on this image: *"Falling back to
  UnquantizedLinearMethod"* (applies only to non-quantized layers — NVFP4 weights are still used) and a
  TokenizersBackend tokenizer warning (no request errors).

- **Qwen3.6-35B-A3B (MoE+GDN) on SGLang — BLOCKED by a GatedDeltaNet block-FP8 shape wall** (measured
  2026-06-23, same nightly image). The arch loads, but `nvidia/Qwen3.6-35B-A3B-NVFP4` crashes during model
  construction inside the hybrid linear-attention layer: `qwen3_5.py Qwen3_5GatedDeltaNet.create_ba_proj`
  builds the GDN b/a gate as a `MergedColumnParallelLinear` with **output_partition_size = 32**, SGLang
  routes it through the **FP8 block-quant** path (`fp8.py validate_block_quant_shapes`, `block_n = 128`),
  and raises `ValueError: Weight output_partition_size = 32 is not divisible by weight quantization
  block_n = 128`. The 32-wide GDN gate isn't block-128 tileable. **This is the answer to the ModelOpt
  open question, but the wall is the hybrid-GDN layer, not NVFP4 packing per se** — nvidia's ModelOpt
  checkpoint quantizes that small GDN projection in FP8-block, which SGLang's qwen3_5 impl can't validate.
  The 27B sibling avoids it because `unsloth/Qwen3.6-27B-NVFP4` packs the GDN gates differently (not
  block-FP8). **No trusted alternative NVFP4 exists for the 35B-A3B** (HF has only `unsloth/...-GGUF`
  → llama.cpp, and an untrusted `dealignai/...-MXFP4-CRACK` repo), so there is no SGLang-viable fallback.
  **Use vLLM for the 35B-A3B NVFP4** (base 430 / +MTP 541 tok/s — both done). Blocks both
  `qwen3-6-35b-a3b-nvfp4-sglang` and its `-mtp` sibling (load fails before MTP/NEXTN is reached).

## Quant notes

- **NVFP4 (W4A4) is a genuine GB10 fast-path even for tiny models** — cosmicproc's `gemma-4-E4B-it-NVFP4`
  (ModelOpt, `FlashInferCutlassNvFp4LinearKernel`) beat FP8 on the same E4B by ~23% decode (the
  benchmark's fastest decode). Worth trying NVFP4 alongside FP8 wherever a trusted NVFP4 exists. (Trust:
  cosmicproc is an *individual* uploader — normally a BLOCK per the trusted-repo policy, run here on the
  user's explicit request; 61k dl/mo + a clean error-free result corroborated it.)

## Spec-decode acceptance — gpt-oss EAGLE3 on ShareGPT = poor & concurrency-degrading
The RedHatAI 20b / NVIDIA-throughput 120b EAGLE3 heads land **mean accept-len only ~1.0–1.7** (avg
draft accept ~1.5–25%) on the ShareGPT + harmony-reasoning stream — far below the ~3.0/~70% expectation
— and it **falls monotonically as concurrency/model-size rises** (20b ~1.7→1.2 over conc 1→8; 120b
~1.25→~1.05/~zero), the *opposite* of the constant-with-concurrency rule. Off-distribution draft tokens
also intermittently **corrupt the harmony channel** → `HarmonyError: unexpected tokens remaining in
message header` / `Unexpected token 0 while expecting start token 200006` (a handful of failed requests,
worse at higher conc). So the published gpt-oss-20b conc-32 "+28% decode" win is a scheduling/prefill
effect at that one batch size, NOT high acceptance — don't generalize it down the sweep, and the 120b
gains nothing at any concurrency. **Takeaway:** these EAGLE3 heads only pay off on **coding** workloads
(where acceptance is high); for general chat, expect spec-decode to be neutral or negative on gpt-oss.
Re-test on a code dataset before claiming an EAGLE3 win for any reasoning model.
