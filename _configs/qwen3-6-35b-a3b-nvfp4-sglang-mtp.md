---
title: Qwen3.6-35B-A3B · SGLang · NVFP4 + MTP
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: SGLang
speculative: MTP (NEXTN)
quant: NVFP4
quant_rationale: NVIDIA's official NVFP4 (nvidia/Qwen3.6-35B-A3B-NVFP4, ModelOpt v0.44.0) + its in-repo MTP module via SGLang's NEXTN path. Preferred over unsloth per policy. NOTE — nvidia documents only a vLLM path, so SGLang support for the ModelOpt NVFP4 format must be verified at run time.
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
spec_acceptance:
completed_at: 2026-06-23 13:08 +08
engine_image: lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed@sha256:ca580c17cf5f9d2e268f4153d977e3cd46528feb2c62a4de8683a05d08da3cf2
run_command: |
  # BLOCKED — same GatedDeltaNet block-FP8 shape wall as the base sibling; load crashes before MTP/NEXTN
  # is reached, so spec-decode is moot.
  SGLANG_IMAGE=lmsysorg/sglang:nightly-dev-cu13-20260623-ba9d5aed \
    scripts/bench-sglang-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 \
    --trust-remote-code --speculative-algo NEXTN --speculative-num-steps 3 \
    --speculative-eagle-topk 1 --speculative-num-draft-tokens 4
  #   ValueError: Weight output_partition_size = 32 is not divisible by weight quantization block_n = 128
  #   (qwen3_5.py create_ba_proj -> fp8.py validate_block_quant_shapes). See the base sibling for the full trace.
---

**BLOCKED — same root cause as the base SGLang sibling.** Load crashes inside SGLang's qwen3_5
GatedDeltaNet (`create_ba_proj`, output_partition_size = 32 not divisible by FP8 block_n = 128) **before
MTP/NEXTN is ever set up**, so the spec-decode path can't be exercised. The `{vLLM, SGLang} × {base, MTP}`
grid for this model is therefore `{vLLM only}`.

- **Failure:** identical `ValueError: Weight output_partition_size = 32 is not divisible by weight
  quantization block_n = 128` — full trace and analysis in the base config `qwen3-6-35b-a3b-nvfp4-sglang`
  and in `notes/INCOMPATIBILITIES.md`. No trusted alternative NVFP4 packing exists for the 35B-A3B.
- **Use the vLLM MTP path instead** — `qwen3-6-35b-a3b-nvfp4-vllm-mtp` (done): decode **541 tok/s** agg
  (conc-32), accept-len ~3.0, with conc-8 / conc-1 siblings also done. That is the supported NVFP4+MTP
  measurement for this model.
