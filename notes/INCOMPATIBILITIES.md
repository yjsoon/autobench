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

**Vocab-dir gotcha (2026-07-01):** the wrapper's historical `~/models/tiktoken_cache` came up **empty and
root-owned** (host user can't reseed it), so gpt-oss died at load with `HarmonyError: invalid tiktoken vocab
file: No such file or directory`. Fix: `bench-vllm-serving.sh` now takes a **`VOCAB_DIR=` override** — the
verified vocab lives at `~/tiktoken_encodings/o200k_base.tiktoken` (sha256 `446a9538…`), so run gpt-oss with
`VOCAB_DIR=$HOME/tiktoken_encodings`. (Root-owned dirs are pervasive here — `~/.cache/huggingface`, `~/models`
— because containers run as root; see BENCHMARKING.md "Model downloads".)

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
- **GENERALIZES to Qwen3.6-35B-A3B (the SAME wall on a current model).** `AEON-7/Ornith-1.0-35B` (Qwen3.6-35B-A3B
  base, hybrid GDN+full-attn, NVFP4) on the **AEON `aeon-vllm-ultimate` image** (vLLM `0.23.0+aeon.sm121a.dflash`)
  hits the identical `assert new_spec.page_size_bytes == max_page_size` (`kv_cache_utils.py:1064`) the instant a
  DFlash draft is added — base serves fine (decode 422 tok/s conc-32, 37.7 tok/s conc-1 @ 256K), +draft asserts at
  `_init_minimal_kv_cache_for_profiling`. Measured 2026-06-28, full trace in `ornith-1-0-35b-aeon-vllm-nvfp4-dflash-blocked`.
  So this is **not a 3.5-only / int4-only quirk** — it's the hybrid+spec KV wall, independent of model gen, quant
  (int4 vs NVFP4), and engine build (stock nightly vs AEON fork).
- **The small-page revision pin clears the page-size assert AND runs end-to-end — CORRECTED 2026-06-28 by direct
  measurement.** The assert is driven by the *draft's* attention page size (sliding_window × num_kv_heads): drafters with
  `sw 4096 / 8 kv-heads` (the 122B's `bce6f76`; `z-lab/Qwen3.6-35B-A3B-DFlash` **`main`** after its "Modal retrain" — the
  rev AEON's card *requires*) double the page and won't unify; the older small-page arch (`sw 2048 / 4 kv-heads` for the
  122B `6c7242c`; `sw None / 4 kv-heads` at the pre-retrain revs `31977fbe13a8` / `f98dc5c2908b`, **HF-API verified**
  `num_key_value_heads: 4`, no sliding window) pads to an equal page. On the **hybrid-GDN** Qwen3.6-35B-A3B target this
  pin was **measured to boot AND serve cleanly** (heretic-NVFP4 and official `nvidia/Qwen3.6-35B-A3B-NVFP4`, AEON
  `aeon-vllm-ultimate` image, ctx 40960, n_spec 11, **0 errors** across conc 1/8/32 with prefix-caching ON) — see
  `qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash` / `qwen3-6-35b-a3b-heretic-aeon-vllm-ultimate-dflash`. So this is a
  **real end-to-end unblock**, not just an assert dodge. **Note the inversion vs AEON's card:** the card *requires* the
  post-2026-04-19 `main` drafter, but THAT one asserts on this box; only the *forbidden* pre-retrain small-page rev boots.
- **CORRECTION: external DFlash on a GDN-hybrid target is NOT architecturally blocked on this box — the earlier
  "blocked four ways / #39273 makes it impossible" claim was WRONG (refuted by measurement 2026-06-28).** The predicted
  failure modes did **not** occur on the AEON fork (`0.23.0+aeon.sm121a.dflash`): no GDN-rollback crash (**#39273**), no
  DGX-Spark prefix-cache IndexError (**#41884** — ran with `--enable-prefix-caching`, 0 errors), and no NVFP4 acceptance
  collapse (**Vassallo nose-dive overstated**: measured ~22–30% avg draft acceptance / mean accept-len ~3.4–4.3 of 11,
  and the *more* FP8-quantized official checkpoint drafted *slightly better* than the BF16-gate heretic, since the drafter
  matches the true base). AEON evidently patched the hybrid-spec path in their fork (PR #40898 SWA etc.); what's "open
  upstream" (#43626/#41884/#41190/#46105) is **stock-vLLM**, not this image.
- **The real reason to keep MTP is throughput economics, not a block.** DFlash on the official checkpoint won only at
  **conc-1 (+8.5%: 101.9 vs 93.9 tok/s)** and **lost under load (conc-8 −7%, conc-32 −26%)**. Cause: MTP drafts 3 tokens
  at **~66% acceptance, accept-len ~3.0-of-3** (almost no wasted compute); DFlash drafts 11 at **~25% acceptance,
  accept-len ~3.7-of-11** (≈7 wasted forward passes/step). On this compute-bound MoE that wasted draft compute sinks
  aggregate throughput exactly when concurrency rises. **Bottom line: keep MTP — it's structurally draft-efficient,
  needs no external drafter, no forbidden drafter revision, and no untrusted image replacing the pinned vLLM. DFlash
  *works* here; it just isn't worth it for a mixed-workload gateway.**

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

- **gpt-oss-120b EAGLE3 + conc-32 streaming = ~70% `Connection reset` on `sglang:spark`; FIXED on nightly.**
  The LMSYS draft (`lmsys/EAGLE3-gpt-oss-120b-bf16`) serves gpt-oss-120b EAGLE3 fine at **conc-1** on the
  `spark` image (0 errors), but at **conc-32** the `spark` image **reproducibly drops ~70% of requests**
  with client-side `Connection reset by peer` / `Broken pipe` (298/1000 completed, 702 errors — stable
  across 3 runs, box idle). **Not** OOM/CUDA/harmony/draft: server logs are clean (no traceback), single
  requests succeed, and draft acceptance stays healthy (~2.2) throughout — the scheduler keeps decoding
  the survivors. EAGLE3 **disables SGLang's overlap scheduler** (`Overlap scheduler is disabled because
  of using eagle3`), and the `spark` build's serving path can't sustain 32 concurrent streams in that
  mode. **Fix: use `lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed` (`SGLANG_IMAGE=` override) — 0
  errors, decode 171.86 tok/s (`gpt-oss-120b-sglang-mxfp4-eagle3-c32`).** Same image family that fixed
  the Qwen3.6 arch (above). Base (no-spec) gpt-oss-120b conc-32 is unaffected on `spark` (overlap
  scheduler stays on).

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

**UPDATE (2026-06-23) — the failure was the DRAFT, not gpt-oss EAGLE3 per se. The SGLang-native LMSYS
draft REVERSES it on the same ShareGPT+harmony workload.** Swapping `nvidia/gpt-oss-120b-Eagle3-throughput`
(vLLM, throughput-tuned, off-distribution for chat) for **`lmsys/EAGLE3-gpt-oss-120b-bf16`** (SGLang's
own SpecForge draft, its documented gpt-oss recipe) lifts mean accept-len from **~1.05–1.25 → ~2.2–2.4**
and avg acceptance from **~9% → ~55–60%**, **concurrency-stable** (conc-1 ~2.4, conc-32 ~2.25 — the
textbook EAGLE3 pattern, not the NVIDIA draft's monotonic collapse). Result: **net speed wins** — conc-1
decode **40.6 tok/s** (vs vLLM/NVIDIA 14.7) with **0 harmony errors** (vs intermittent corruption), and
conc-32 decode **171.86 tok/s = +22% over base SGLang (140.3)**, the first gpt-oss-120b spec-decode win
here. So: "EAGLE3 is useless on gpt-oss general chat" was really **"the NVIDIA throughput draft is
off-distribution"** — a *workload-matched* draft (LMSYS/SpecForge) pays off even on ShareGPT. Always
prefer the engine's own recipe draft. See `gpt-oss-120b-sglang-mxfp4-eagle3-c1` / `-c32`. (Caveat: needs
the SGLang nightly for conc-32 — the spark image's eagle3 serving bug, above.)

**UPDATE (2026-07-01) — gpt-oss-20b EAGLE3 acceptance is CONCURRENCY-DRIVEN (rises with batch), reversing the
"~5% / concurrency-degrading" reading above.** The old low numbers came from **cap-truncated low-conc runs**.
Fresh vLLM sweep (RedHatAI draft, cu130-nightly, ShareGPT): avg draft acceptance **~5% at conc 2/4/8** but
**~44% (mean accept-len ~2.3) at conc 16/32** — and decode flips from **−29% (c2) / −33% (c4)** vs base to
**+27% (c16) / +28% (c32)**. **Controlled diagnostic proves it's concurrency, not sampling:** conc-16
restricted to the *same first ~150 prompts* as the conc-2 run still gives ~44% (vs ~5% at conc-2 on those very
prompts). So the conc-32 "+28%" is **acceptance-backed, not a scheduling artifact** — and the rule-of-thumb
"acceptance is workload-driven, not concurrency-driven" **fails for this vLLM EAGLE3 path** (draft is
under-accepted at small batch — likely a CUDA-graph/scheduler batch-size effect; mechanism unconfirmed).
Net: EAGLE3 on gpt-oss-20b is a loss below ~conc-8 and a real win at conc ≥16. Pages
`gpt-oss-20b-vllm-mxfp4-eagle3-c{2,4,16}` + matched bases `-mxfp4-c{2,4,16}`.
