---
title: Qwen3.6-35B-A3B · SGLang · NVFP4
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: SGLang
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) on SGLang — preferred over unsloth per policy (use the nvidia image when one exists). Base (non-speculative); MTP in the -mtp sibling. Cross-engine compare vs the vLLM NVFP4 run. NOTE — nvidia documents only a vLLM path, so SGLang support for the ModelOpt NVFP4 format must be verified at run time.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, conc-32]
status: blocked
prefill_toks:
decode_toks:
mem_gb:
mem_source:
completed_at: 2026-06-23 13:08 +08
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # BLOCKED — SGLang's qwen3_5 GatedDeltaNet (GDN) layer fails block-FP8 shape validation on the
  # nvidia ModelOpt 35B-A3B checkpoint. Not a download/arch problem (transformers 5.8.1 loads the arch fine):
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
    scripts/bench-sglang-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 --trust-remote-code
  # Crash during load (qwen3_5.py create_ba_proj -> MergedColumnParallelLinear -> fp8.py
  # validate_block_quant_shapes):
  #   ValueError: Weight output_partition_size = 32 is not divisible by weight quantization block_n = 128.
---

**BLOCKED — SGLang's qwen3_5 GatedDeltaNet can't load the nvidia ModelOpt 35B-A3B NVFP4.** The arch loads
(transformers 5.8.1 on the `nightly-dev-cu13-20260623` image — same image the 27B sibling runs on), but
load crashes inside the hybrid linear-attention layer, not on the MoE NVFP4 weights.

- **Exact failure:** `qwen3_5.py` → `Qwen3_5GatedDeltaNet.create_ba_proj` builds the GDN b/a gate as a
  `MergedColumnParallelLinear` with **output_partition_size = 32**. SGLang routes it through the **FP8
  block-quant** path (`fp8.py validate_block_quant_shapes`, `block_n = 128`) and raises
  `ValueError: Weight output_partition_size = 32 is not divisible by weight quantization block_n = 128`.
  The 32-wide GDN gate projection simply isn't block-128 tileable.
- **Independent of the MoE NVFP4 quant** — this is a SGLang hybrid-GDN + block-FP8 shape wall, tripped by
  the ModelOpt checkpoint quantizing the small GDN projections in FP8-block. It is **not** a
  ModelOpt-NVFP4-packing rejection (the risk this config originally flagged).
- **Why the 27B succeeded but this doesn't:** the 27B uses `unsloth/Qwen3.6-27B-NVFP4` (different packing,
  GDN gates not block-FP8). **No trusted unsloth/nvidia NVFP4 exists for the 35B-A3B** — HF has only
  `unsloth/...-GGUF` (llama.cpp path, not SGLang) and an untrusted `dealignai/...-MXFP4-CRACK` repo
  (skipped on trust grounds). So there is no SGLang-viable NVFP4 fallback for this model.
- **Use the vLLM NVFP4 path instead** — `qwen3-6-35b-a3b-nvfp4-vllm` (done, 430 tok/s base / 541 tok/s
  +MTP) is the supported route. See `notes/INCOMPATIBILITIES.md`.
- **Spec-decode sibling** `qwen3-6-35b-a3b-nvfp4-sglang-mtp` is blocked for the same reason (load fails
  before MTP is even reached).
