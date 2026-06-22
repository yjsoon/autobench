---
title: DeepSeek V4-Flash · llama.cpp · IQ2_XXS-XL (GGUF)
model: deepseek-ai/DeepSeek-V4-Flash
company: DeepSeek
family: DeepSeek
params: 284B (MoE) — measured (the other DeepSeek-V4-Flash pages' "158B" is wrong)
engine: llama.cpp
quant: IQ2_XXS-XL
quant_rationale: The ONLY size class that fits one Spark. DeepSeek-V4-Flash is ~284B params (not 158B) — NVFP4 is 168 GB, Q8 ~290 GB, Q4 ~175 GB, all over the 121 GB ceiling; only the aggressive IQ2/Q2 GGUFs fit (IQ2_XXS-XL ~79 GB). Quality is degraded but the benchmark measures throughput. teamblobfish GGUF (56k dl/mo) on a deepseek_v4-capable llama.cpp fork.
source_repo: teamblobfish/DeepSeek-V4-Flash-GGUF
download_url: https://huggingface.co/teamblobfish/DeepSeek-V4-Flash-GGUF
context: 8192
modalities: [text]
mm_served: false
concurrency: 8
tags: [deepseek-v4-flash, DeepSeek, IQ2_XXS-XL, 130B+, conc-8]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
measured_on: 2026-06-23
completed_at:
engine_image: autobench-llamacpp-v4flash (locally built — cchuter/llama.cpp @ feat/v4-port-cuda, commit 781e978, CUDA sm_121; Dockerfile scripts/Dockerfile.llamacpp-v4flash)
run_command: |
  # BLOCKED at graph construction (see Notes). Build the fitting path from source:
  #   docker build -f scripts/Dockerfile.llamacpp-v4flash -t autobench-llamacpp-v4flash .
  #     (FROM nvidia/cuda:13.0.1-devel; clones cchuter/llama.cpp -b feat/v4-port-cuda;
  #      cmake -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=121; links libcuda stub for the VMM calls)
  #   hf download teamblobfish/DeepSeek-V4-Flash-GGUF --include "IQ2_XXS-XL/*"   # ~79 GB, 2 shards
  docker run -d --name llamacpp-dsv4 --gpus all -p 8081:8081 \
    -v ~/.cache/huggingface:/root/.cache/huggingface:ro \
    autobench-llamacpp-v4flash /llama.cpp/build/bin/llama-server \
    -m .../IQ2_XXS-XL/DeepSeek-V4-Flash-IQ2_XXS-XL-00001-of-00002.gguf \
    -ngl 999 -c 8192 --parallel 1 -cb -fa off -fit off -b 4096 -ub 2048 --jinja \
    --host 0.0.0.0 --port 8081
  # → GGML_ASSERT(obj_new) failed (ggml.c:1778) in llama_model_deepseek4::graph during sched_reserve
---

**BLOCKED 2026-06-23 — the model *loads* on a fitting GGUF + matched fork, but the experimental fork's
graph construction aborts on GB10. Pursued at the user's "compile what you need to" request after the
NVFP4 vLLM config proved unfittable.**

**Why this exists:** [`deepseek-v4-flash-vllm-nvfp4-eagle3`] can't fit — DeepSeek-V4-Flash is **~284 B
params** (llama.cpp reports `model params = 284.33 B`; the "158B" on the other pages is wrong), so every
build that preserves quality is over the 121 GB ceiling: NVFP4 **168 GB**, Q8 **~290 GB**, Q4 **~175 GB**.
**Only the aggressive IQ2/Q2 GGUFs fit** — IQ2_XXS-XL is **~79 GB** (2 shards), Q2_K-XL ~107 GB. This is
the fitting path, benchmarked for throughput (IQ2 quality is poor, but speed is the metric).

**What worked:**
- **Built a `deepseek_v4`-capable llama.cpp with CUDA for GB10** (`sm_121`). `deepseek_v4` is **not in
  upstream** llama.cpp — two independent WIP forks exist:
  - ggml-org **PR #22378** (`deepseek4` arch) — built it, but the teamblobfish GGUF wouldn't load:
    `missing tensor 'hc_head_base'` (its tensor layout differs from this fork's).
  - **`cchuter/llama.cpp` `feat/v4-port-cuda`** — the fork the teamblobfish GGUFs are quantized for (5
    custom `dsv4` ops: `rope_tail`, `hc_split_sinkhorn`, `hc_weighted_sum`, `hc_expand`,
    `fp8_kv_quantize`). Built it (`Dockerfile.llamacpp-v4flash`; needed a `-lcuda` stub link flag so
    ggml-cuda's VMM calls resolve). Binary runs, sees **GB10 sm_121, VMM yes**.
- **The model loads with the matched fork:** weights load (284 B, 256 experts), the DeepSeek4 compressed-KV
  buffer allocates, `sched_reserve` resolves "fused Gated Delta Net (autoregressive + chunked) enabled".

**The blocker — graph construction aborts (experimental-fork bug, GB10):** during `sched_reserve` →
`llama_model_deepseek4::graph`, ggml aborts. Two configs, two asserts:
- **conc 8 / ctx 32768:** `GGML_ASSERT(ggml_nelements(a) == ne0*ne1*ne2)` (`ggml.c:3660`) — a reshape in
  the HC-attention graph.
- **conc 1 / ctx 8192** (with and without `-fit off`): `GGML_ASSERT(obj_new)` (`ggml.c:1778`,
  `ggml_new_object`) — graph object-pool allocation fails while building the deepseek4 graph.

Both are in the fork's graph builder, not the config — the fork is explicitly "**not extensively tested**"
(version 1). Resolving them means C++-debugging an experimental fork's custom attention graph (out of
scope, and risks silently-wrong outputs against the trusted-path policy). **Re-test when DeepSeek-V4
support lands in upstream llama.cpp** (PR #22378 / a stable release) with matching upstream GGUFs.

**Net for DeepSeek-V4-Flash on one Spark:** no path runs *today* — NVFP4/FP8/Q4 don't fit (284 B), and the
only fitting quant (IQ2 GGUF) needs an experimental fork whose GB10 graph path is buggy. The infra is
captured (Dockerfile + recipe above) for a clean re-test once upstream support matures.
