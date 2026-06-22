---
title: DeepSeek V4-Flash · vLLM · NVFP4 + EAGLE3.1
model: deepseek-ai/DeepSeek-V4-Flash
company: DeepSeek
family: DeepSeek
params: 158B (MoE)
engine: vLLM
speculative: EAGLE3.1
quant: NVFP4
quant_rationale: NVIDIA's NVFP4 (~79 GB) is the only build of this 158B MoE that fits one Spark — FP8 is ~160 GB. Base nvidia/DeepSeek-V4-Flash-NVFP4 + the ManiacLabs EAGLE3.1 draft (hidden sizes match at 4096). Blocked on engine support — see Notes.
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

**Blocked — got *agonizingly* close: everything works except v0.22.0's NVFP4 fused-MoE kernels lack
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
are saved above and ready. (Not pursued: hand-porting kernels across the 0.19↔0.22 ABI, or rebuilding
flashinfer for sm_121 — both heavy and fragile.)
