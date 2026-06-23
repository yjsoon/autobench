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
status: pending
prefill_toks:
decode_toks:
mem_gb:
mem_source:
spec_acceptance:                 # capture avg draft acceptance + mean acceptance length; cross-check vs FP8+MTP
completed_at:
engine_image: vllm/vllm-openai:nightly-aarch64
run_command: |
  # INTENDED (not yet run). NVIDIA's DGX Spark MTP recipe: base MoE on the marlin backend, MTP head on
  # the triton MoE backend (--moe-backend marlin + moe_backend:triton inside --speculative-config).
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 32 1000 900 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3 --moe-backend marlin \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'
  # = vllm/vllm-openai:nightly-aarch64, --gpu-memory-utilization 0.85 --max-num-seqs 32 (wrapper defaults)
---

**Queued — Qwen3.6-35B-A3B NVFP4 + native MTP on vLLM (NVIDIA official quant + DGX Spark recipe).**
Spec-decode counterpart of the NVFP4 MoE base run.

- **Repo — NVIDIA official:** [`nvidia/Qwen3.6-35B-A3B-NVFP4`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4)
  (ModelOpt v0.44.0). NVIDIA documents the MTP form directly in its DGX Spark vLLM recipe → `Spark recipe`.
- **MoE-specific MTP:** base MoE on `--moe-backend marlin`, MTP head on `moe_backend:triton` inside
  `--speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'`. MTP weights
  ship in-repo.
- **Acceptance:** capture avg draft acceptance + mean acceptance length, cross-check vs the FP8+MTP 35B-A3B
  runs and the 27B NVFP4+MTP run — a large gap flags a quant/MoE-backend interaction.
- **conc-8/conc-1 variants** to be added when this spec config is benchmarked (per the FP8-MTP pattern).
