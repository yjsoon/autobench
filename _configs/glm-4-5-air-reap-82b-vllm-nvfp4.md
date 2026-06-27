---
title: GLM-4.5-Air-REAP 82B · vLLM · NVFP4 (W4A16)
model: cerebras/GLM-4.5-Air-REAP-82B-A12B (NVFP4 W4A16, self-quantized)
company: Cerebras (REAP) · Zhipu AI (base)
family: GLM
params: 82B / 12B (MoE, REAP-pruned from 106B/12B)
engine: vLLM
quant: NVFP4 (W4A16)
quant_rationale: Self-quantized from the Cerebras BF16 REAP checkpoint to compressed-tensors NVFP4A16 (4-bit float E2M1, block-16 fp8_e4m3 scale + fp32 global scale; weight-only). Produced by a custom shard-by-shard streaming quantizer (reusing compressed_tensors' own pack/scale primitives) because ModelOpt OOMs materializing the 159 GB model in 121 GB RAM and LLM Compressor's offload converter is brittle — the streamer peaks at ~one shard of RAM. W4A16 over W4A4 because on GB10 the NVFP4 win is memory-bandwidth via the marlin dequant kernel, not FP4 compute, and W4A16 keeps ~2% accuracy vs W4A4's >4% on an already double-compressed (REAP) model.
source_repo: cerebras/GLM-4.5-Air-REAP-82B-A12B
download_url: https://huggingface.co/gauravmm/GLM-4.5-Air-REAP-82B-A12B-NVFP4
hf_repo: gauravmm/GLM-4.5-Air-REAP-82B-A12B-NVFP4
hf_url: https://huggingface.co/gauravmm/GLM-4.5-Air-REAP-82B-A12B-NVFP4
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [glm-4-5-air-reap-82b, Cerebras, GLM, NVFP4, 41-130B, Spark recipe, REAP, Self-Quantized, conc-32]
status: done
prefill_toks: 163.22
decode_toks: 158.39
mem_gb: 89.2
mem_source: system MemAvailable delta from pre-server idle — vLLM 0.72 reservation (real weights ~51 GB, rest is KV/activations); see Notes
measured_on: 2026-06-26
completed_at: 2026-06-26 16:10 +0800
run_command: |
  # NVFP4A16 checkpoint produced locally by ~/Desktop/reap-nvfp8/llmc/streaming_quantize.py
  # (compressed-tensors nvfp4-pack-quantized), mounted as a LOCAL path (not an HF id).
  # vllm/vllm-openai:nightly-aarch64 ENTRYPOINT is ["vllm","serve"] → pass <model> <flags>.
  # --moe-backend marlin is mandatory; flashinfer_cutlass REJECTS W4A16 NVFP4 and crashes
  # engine-core init on this box. --quantization compressed-tensors reads the quant config.
  # gpu-mem-util 0.72 (not 0.85): a separate ~27 GB baseline (serving-qwen25-coder-7b +
  # jina-embed) was resident; 0.85 risks OOM. 0.72*121 ≈ 87 GB reservation fits cleanly.
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/Desktop/reap-nvfp8/models/GLM-4.5-Air-REAP-82B-A12B-NVFP4A16:/model:ro \
    vllm/vllm-openai:nightly-aarch64 \
    /model \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.72 --max-num-seqs 32 \
    --quantization compressed-tensors --moe-backend marlin --trust-remote-code \
    --served-model-name glm-air-reap-nvfp4
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model glm-air-reap-nvfp4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Done — self-quantized NVFP4A16, serves on vLLM/marlin.** This is the **REAP-pruned** GLM-4.5-Air
(Cerebras `GLM-4.5-Air-REAP-82B-A12B`: 82B total / 12B active, 96 experts, 46 layers + 1 MTP head,
`Glm4MoeForCausalLM`) — distinct from the dense `zai-org/GLM-4.5-Air` stub (still blocked on a quant
choice). REAP is router-weighted expert-activation pruning (~106B → 82B), so this is a *second*
compression stage stacked under our NVFP4. No published NVFP4 of this REAP checkpoint exists, so we
quantized the **BF16** source ourselves (Cerebras recommends BF16 for low-bit — avoids stacking FP8
rounding under FP4) and published it →
**[`gauravmm/GLM-4.5-Air-REAP-82B-A12B-NVFP4`](https://huggingface.co/gauravmm/GLM-4.5-Air-REAP-82B-A12B-NVFP4)**.

**Result (conc-32 ShareGPT, 65536 ctx):** prefill **163 tok/s**, decode **158 tok/s**, ttft_median
623 ms, tpot_median 192 ms, 0 errors (580/1000 prompts in the 900 s cap). On-disk checkpoint 51 GB;
runtime footprint 89 GB = vLLM 0.72 reservation (≈51 GB weights + KV/activations). Decode beats the
120B-class `Nemotron-3-Super-120B-A12B-NVFP4` (97 tok/s) — fewer active params (12B).

**How it was quantized (the interesting part) — `~/Desktop/reap-nvfp8/llmc/streaming_quantize.py`:**
off-the-shelf tools both failed on the 159 GB BF16 model vs 121 GB RAM. **ModelOpt** `mtq.quantize`
eagerly materializes the whole model → OOM (peaked 111 GB, climbing). **LLM Compressor** loads bounded
(disk offload) but compressed_tensors 0.17's `from_accelerate` converter asserts offloaded params are
on `meta` and rejects accelerate's layout. So: a **shard-by-shard streaming quantizer** that never
loads the full model — per Linear weight it reuses compressed_tensors' own primitives (`generate_gparam`
fp32 global scale; `compute_dynamic_scales_and_zp` per-block-16 fp8_e4m3 scale; `pack_fp4_to_uint8`) to
emit byte-compatible `nvfp4-pack-quantized`. Peak RAM ≈ one shard (~5 GB). **Critical correctness fix:**
vLLM FUSES parallel projections (q/k/v→qkv, gate/up→gate_up, expert w1/w3) and forces ONE
`weight_global_scale` per fused group; a per-tensor global scale → coherent-looking GARBAGE output
(vLLM warns *"weight global scale is different for parallel layers"* / *"w1 must match w3"*). Fixed with
a 2-pass run: pass 1 collects per-group min/max, pass 2 quantizes each fused group against a shared
global scale. Must set `TORCH_COMPILE_DISABLE=1` (this box's gcc can't build triton/inductor's CUDA util).

**Serve recipe (GB10):** `--quantization compressed-tensors --moe-backend marlin --trust-remote-code`.
marlin MoE is **mandatory** — flashinfer_cutlass rejects W4A16 NVFP4 and crashes engine-core init
(matches the `Qwen3.6-35B-A3B-NVFP4` findings in INCOMPATIBILITIES). `nightly-aarch64` loaded `glm4_moe`
fine (the NVFP4-loading regression there is Gemma-4-specific).
