---
title: DeepSeek V4-Flash · vLLM · NVFP4 + EAGLE3.1
model: deepseek-ai/DeepSeek-V4-Flash
company: DeepSeek
family: DeepSeek
params: 158B (MoE)
engine: vLLM
speculative: EAGLE3.1
quant: NVFP4
quant_rationale: Intended as NVIDIA's NVFP4 base + ManiacLabs EAGLE3.1 draft (hidden sizes match at 4096). BLOCKED ON FIT, not engine — the repo's *actual* weights are 168.3 GB (MIXED_PRECISION = FP8 backbone + NVFP4 experts; the "~79 GB" was a nominal-4-bit estimate that never held), over the 121 GB ceiling. See Notes.
source_repo: nvidia/DeepSeek-V4-Flash-NVFP4
download_url: https://huggingface.co/ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [deepseek-v4-flash, DeepSeek, NVFP4, 130B+, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on:
completed_at:
run_command: |
  # Recipe is correct & ready (verified to the kernel-select step); blocked only by a GB10 NVFP4-MoE
  # kernel gap in vLLM 0.22.0 (see Notes). Dockerfile: scripts/Dockerfile.vllm-v4flash
  #   FROM vllm/vllm-openai:v0.22.0                 # natively bundles vllm/models/deepseek_v4 + eagle3
  #   ENV VLLM_ALLOW_INSECURE_SERIALIZATION=1  EAGLE3_DRAFT_KV_CACHE_DTYPE=auto
  # Run (single Spark, adapted from ManiacLabs SERVING.md's 4-GPU recipe → TP=1, NVFP4 base to fit):
  #   docker run --gpus all --ipc=host -p 8000:8000 \
  #     -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
  #     autobench-vllm-v4flash nvidia/DeepSeek-V4-Flash-NVFP4 \
  #     --max-model-len 8192 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
  #     --trust-remote-code --enforce-eager --kv-cache-dtype fp8 --tensor-parallel-size 1 \
  #     --speculative-config '{"method":"eagle3","model":"ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1","num_speculative_tokens":3}'
---

## CORRECTION 2026-06-23 — the real blocker is FIT (168 GB), not the engine. The engine arrived; the model still doesn't fit.

**The earlier "NVFP4 ~79 GB fits / blocked on a GB10 NVFP4-MoE kernel gap" conclusion was wrong on the
decisive fact.** Re-checked the actual downloaded weights: `nvidia/DeepSeek-V4-Flash-NVFP4`'s own
`safetensors.index.json` reports **`total_size = 168.3 GB`** (157 GiB on disk, 46 shards). Its
`config.json` is **`quant_algo: MIXED_PRECISION` / `quant_method: fp8`** with per-layer NVFP4 only on the
experts — i.e. an FP8 backbone + NVFP4 experts, essentially **FP8-sized**, not the nominal ~79 GB a pure
4-bit 158B would be. **168 GB > the 121 GB unified-memory ceiling**, so the weights alone can't be held,
let alone KV + the EAGLE3.1 draft. This was never caught before because the model failed at the kernel
step *before* it ever tried to allocate weights. **No engine/kernel/oracle/compilation work changes this
— it's a hard fit wall.**

**Engine support, meanwhile, has fully arrived (now moot for this build):** `vllm/vllm-openai:nightly-aarch64`
(vLLM **0.23.1rc1**, the cu130-nightly successor — cu130-nightly itself is **abandoned**, no push in
~2 months) ships **`DeepseekV4ForCausalLM`** + **`deepseek_eagle3.py`** + a **rewritten NVFP4-MoE oracle**
(CUTLASS/CuteDSL/Marlin fallbacks, no blanket sm_121 reject). `tie_word_embeddings` is False, so it would
also dodge the gemma `tie_weights` regression. So the two-image standoff below is **resolved** — one image
now has both halves — but it doesn't matter, because the NVFP4 build is 168 GB.

**To actually benchmark DeepSeek-V4-Flash on one Spark you need a ~80 GB build (none is NVFP4):**
- **GGUF Q4_K_M / IQ-quants (~80 GB)** — fits, but `deepseek_v4` is **not in upstream llama.cpp**; only
  community/WIP forks load it (antirez's CPU+Metal experimental fork; the `ggml-org` `wip/deepseek-v4-support`
  branch / PR #22378 with CUDA-Blackwell work). Requires building a WIP fork from source.
- **INT4-AWQ/GPTQ (~80 GB) on vLLM** — fits, but only from community (individual-uploader) repos → trust
  per the repo policy.

So the **`vllm-nvfp4-eagle3` config as specified is physically impossible on a single 121 GB Spark.** Status
stays **blocked**; the unblock is a fitting 4-bit build (different quant, and for GGUF a different engine),
not a compile. Left for a follow-up pending an explicit choice of fitting path.

<details><summary>Superseded 2026-06-22 investigation (engine-support angle — kept for the record)</summary>

**Got *agonizingly* close: everything works except v0.22.0's NVFP4 fused-MoE kernels lack
GB10 support.** Deep investigation 2026-06-22 at the user's request (try a newer engine, keep it in
Docker, capture a full recipe).

**Everything that works (verified, in order):**
- **Engine arch:** the stub's premise ("no engine has `deepseek_v4`") is **outdated for vLLM 0.22.0** —
  it **natively bundles `vllm/models/deepseek_v4`** (the ManiacLabs SERVING.md overlay was upstreamed;
  that overlay's GitHub repo is now 404, and doesn't matter). transformers 5.9.0 recognizes
  `deepseek_v4`; the registry has `DeepSeekV4MTP` + `deepseek_eagle3`.
- **Base fits:** `nvidia/DeepSeek-V4-Flash-NVFP4` (~79 GB) vs the 160 GB FP8 — NVIDIA-official, fits one
  Spark.
- **Draft compatible:** `ManiacLabs/DeepSeek-V4-Flash-EAGLE3.1` is a standard `LlamaForCausalLMEagle3`
  head, `target_hidden_size` 4096 = base `hidden_size` 4096.
- **GB10 runs v0.22.0:** smoke test passed — `torch 2.11.0+cu130`, device **NVIDIA GB10**, capability
  **(12, 1) = sm_121**, a real matmul executed; and at load vLLM selected `CutlassFp8BlockScaledMMKernel`
  for the FP8 linear layers (so **FP8 kernels work on GB10 in 0.22.0**).

**The one hard blocker — a 4-bit-MoE kernel gap (not the model, not the arch, not GB10 generally):**
```
NotImplementedError: No NvFp4 MoE backend supports the deployment configuration.
DEBUG  NvFp4 MoE backend 'FLASHINFER_TRTLLM' does not support the deployment
       configuration since kernel does not support current device cuda.
```
v0.22.0 ships **flashinfer 0.6.11**, whose **NVFP4 fused-MoE kernels have no GB10/sm_121 build** — every
NVFP4 MoE backend (FlashInfer-TRTLLM, CUTLASS, Marlin) reports the device unsupported. It's specifically
the **4-bit NVFP4 expert** path; FP8 linear is fine. Forcing a backend
(`VLLM_USE_FLASHINFER_MOE_FP4=1`) hits the same wall.

**A genuine two-image standoff — neither has both halves:**

| Image | `deepseek_v4` model | GB10 NVFP4-MoE kernels |
|---|---|---|
| `vllm/vllm-openai:v0.22.0` | ✅ native | ❌ (flashinfer 0.6.11, no sm_121) |
| `vllm/vllm-openai:cu130-nightly` | ❌ (vLLM 0.19.2) | ✅ (the Nemotron NVFP4 MoEs run here) |

**Unblock path (clean, no surgery):** the `cu130-nightly` tag is rolling — **once it advances to
≥ 0.22, it has *both*** the `deepseek_v4` model and the GB10 NVFP4-MoE kernels, and the recipe above
runs as-is. Alternatively, a v0.22.x image rebuilt with a GB10-capable flashinfer (≥ the version in
cu130-nightly). The Dockerfile (`scripts/Dockerfile.vllm-v4flash`) and the exact single-Spark command
are saved above and ready.

---

**Update — tried upgrading vLLM *inside* cu130-nightly (`scripts/Dockerfile.vllm-cu130-022`). Got 2 of
3 barriers down; the 3rd is a vLLM-0.22 code gate, not a kernel gap.**
- `pip install vllm==0.22.0` inside cu130-nightly **works**: there's an `aarch64` wheel, and its
  **`vllm._C` runs on GB10** (rms_norm kernel verified on sm_121) — so no source rebuild is needed, and
  `deepseek_v4` is now present on a GB10-capable image. ✅✅
- The dep bump pushed flashinfer 0.6.8→0.6.11, but the **`flashinfer-jit-cache` stays `0.6.8.post1+cu130`**
  (the GB10 kernels). Pinning `flashinfer-python==0.6.8.post1` (+ `FLASHINFER_DISABLE_VERSION_CHECK=1`)
  re-matches them.
- **Still blocked at the same line:** vLLM **0.22.0's** NVFP4-MoE *oracle*
  (`fused_moe/oracle/nvfp4.py::select_nvfp4_moe_backend`) raises *No NvFp4 MoE backend supports the
  deployment configuration* — its `is_supported_config` checks **gate out sm_121 for every backend**
  (FlashInfer-TRTLLM / CUTLASS / Marlin) regardless of the cu130 kernels being present. vLLM **0.19.2's**
  older NVFP4-MoE path *did* accept GB10 (the Nemotron NVFP4 MoEs run on cu130-nightly), but it has no
  `deepseek_v4`. So the real gap is **a GB10 capability gate in vLLM 0.22's MoE oracle code**, which only
  a proper ≥0.22 GB10 build (or a targeted oracle patch — vLLM-internals surgery with crash risk if the
  forced backend's kernel doesn't actually fit) would resolve. Conclusion stands: **wait for
  cu130-nightly ≥ 0.22**, then the saved recipe runs as-is.

*(Superseded: nightly-aarch64 = vLLM 0.23.1 now has both `deepseek_v4` and a GB10-capable NVFP4-MoE oracle,
so this engine angle is resolved — but the 168 GB fit wall above makes it moot.)*

</details>
