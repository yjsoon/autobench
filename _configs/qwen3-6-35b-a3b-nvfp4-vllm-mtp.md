---
title: Qwen3.6-35B-A3B · vLLM · NVFP4 + MTP
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
speculative: MTP
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) + the checkpoint's own MTP module — NVIDIA's DGX Spark recipe gives the exact MTP form. Preferred over unsloth per policy (use the nvidia image when one exists). Native multi-token-prediction, no separate draft.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-32]
status: done
prefill_toks: 557.59
decode_toks: 541.26
mem_gb: 110.28
mem_source: system MemAvailable delta (10s sampling) — NVFP4 MoE + in-repo MTP head
spec_acceptance: mean acceptance length ~3.0 · avg draft acceptance ~66–69% · per-position 0.85/0.68/0.53 (3 spec tokens)
measured_on: 2026-06-23
completed_at: 2026-06-23 12:34 +08
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # NVIDIA's DGX Spark MTP recipe: base MoE on the marlin backend, MTP head on the triton MoE
  # backend (--moe-backend marlin + moe_backend:triton inside --speculative-config).
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3 --moe-backend marlin \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 32 (wrapper defaults).
  # 1000/1000 prompts, 0 errors, finished in 472.7 s (did NOT hit the 900 s cap). ready after 394 s.
  # TPOT median reads 0.0 — the qwen3 reasoning-parser splits reasoning/content so the client TPOT is
  # unreliable here (same caveat as the base 35B-A3B run); trust the aggregate decode tok/s + SpecDecoding log.
---

**DONE — fastest decode in the benchmark so far (541 tok/s).** Qwen3.6-35B-A3B NVFP4 + native MTP on
vLLM, NVIDIA official quant + DGX Spark recipe. Spec-decode counterpart of the NVFP4 MoE base run.

- **Result (conc 32):** prefill 557.6 / decode **541.26** tok/s aggregate; **1000/1000, 0 errors**,
  finished in **472.7 s (well under the 900 s cap)** — a full-coverage measurement, not a cap-truncated
  one. Peak mem **110.28 GB**.
- **MTP speedup vs base:** decode **541.26 (MTP) vs 430.76 (base)** = **+26%**. The NVFP4 MoE fast-path
  (+50% vs FP8) and MTP (+26%) compound — this NVFP4+MTP point is **~2.4× the FP8 base** (285.97).
- **Acceptance:** mean acceptance length **~3.0** (3 spec tokens), avg draft acceptance **~66–69%**,
  per-position **0.85 / 0.68 / 0.53** — healthy and stable across the run. Matches the 27B NVFP4+MTP
  (~3.0) and the SGLang NEXTN run (~3.0) — **the in-repo MTP module drafts consistently across model size
  and engine**; the throughput differences are engine/scheduler, not draft quality.
- **MoE-specific MTP works:** base MoE on `--moe-backend marlin`, MTP head on `moe_backend:triton` (inside
  `--speculative-config`) loaded and ran with **0 errors** — the marlin-base / triton-MTP split is the
  correct GB10 recipe for this checkpoint. Log confirms weight-sharing (MTP shares target embedding +
  lm_head) and the hybrid attention/`mamba` page-size reconciliation (GDN layers present).
- **"NVFP4" here is really W4A16 (weight-only) — there is no native-FP4 *compute* to capture.** The
  checkpoint quantizes the experts to **4-bit weights but keeps 16-bit activations** (`quant_algo
  W4A16_NVFP4`). Native FP4 tensor-core math needs **W4A4** (FP4 activations too); with bf16 activations
  the matmul *must* dequantize the weights — exactly what marlin does. `--moe-backend flashinfer_cutlass`
  (the native sm_121 FP4 GEMM) **rejects this scheme outright** (`NvFp4 MoE backend 'FLASHINFER_CUTLASS'
  does not support … quantization scheme QuantKey(u8, scale(f8e4m3fn,…GroupShape(row=1,col=16)), …)`),
  verified on GB10 — engine-core init fails. So **marlin is the correct kernel, not a fallback**, and the
  NVFP4 win is **memory bandwidth** (smaller expert weights → less MoE traffic → faster decode), *not* FP4
  tensor cores. A real FP4-compute speedup would require a W4A4 NVFP4 export of this model.
- **TPOT caveat:** client TPOT median reads 0.0 because the `qwen3` reasoning-parser splits
  reasoning/content streams (same as the base run) — trust the aggregate decode tok/s and the in-engine
  SpecDecoding metrics, not the client TPOT.
- **Repo — NVIDIA official:** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
  (ModelOpt v0.44.0); MTP form from NVIDIA's DGX Spark vLLM recipe → `Spark recipe`.
- **conc-8 / conc-1 variants:** queued next (`-c8`, `-c1`) per the FP8-MTP pattern.
