---
title: MiniMax-M2.5-REAP 139B · vLLM · NVFP4 (W4A16)
model: cerebras/MiniMax-M2.5-REAP-139B-A10B (NVFP4 W4A16, self-quantized)
company: Cerebras (REAP) · MiniMax (base)
family: MiniMax
params: 139B / 10B (MoE, REAP-pruned from 230B/10B — 256→154 experts)
engine: vLLM
quant: NVFP4 (W4A16)
quant_rationale: Self-quantized from the Cerebras block-FP8 REAP checkpoint to compressed-tensors NVFP4A16 (4-bit float E2M1, block-16 fp8_e4m3 scale + fp32 global scale; weight-only). No BF16/FP16 REAP exists, so the source is block-FP8 (float8_e4m3fn + weight_scale_inv [128,128]); the streaming quantizer dequantizes FP8→BF16 on the fly per shard (no ~280 GB BF16 intermediate) before NVFP4. Produced by the same shard-by-shard streaming quantizer as the GLM-Air REAP run because ModelOpt OOMs materializing the model in 121 GB RAM and LLM Compressor's offload converter is brittle — the streamer peaks at ~one shard of RAM. W4A16 over W4A4 because on GB10 the NVFP4 win is memory-bandwidth via the marlin dequant kernel, not FP4 compute, and W4A16 keeps ~2% accuracy vs W4A4's >4% on an already double-compressed (REAP) model.
source_repo: cerebras/MiniMax-M2.5-REAP-139B-A10B
download_url: https://huggingface.co/cerebras/MiniMax-M2.5-REAP-139B-A10B
hf_repo: gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4
hf_url: https://huggingface.co/gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4
context: 65536
modalities: [text]
mm_served: true
concurrency: 32
tags: [minimax-m25-reap-139b, Cerebras, MiniMax, NVFP4, 131-260B, Spark recipe, conc-32]
status: done
prefill_toks: 127.93
decode_toks: 120.0
mem_gb: 109
mem_source: system MemAvailable while serving (used 109/121 GB) — vLLM 0.85 reservation (real weights ~75 GB, rest is KV/activations/CUDA); see Notes
measured_on: 2026-06-26
completed_at: 2026-06-26 22:25 +0800
run_command: |
  # NVFP4A16 checkpoint produced locally by ~/Desktop/reap-nvfp8/llmc/streaming_quantize.py
  # (compressed-tensors nvfp4-pack-quantized), mounted as a LOCAL path (not an HF id).
  # vLLM resolves MiniMaxM2ForCausalLM NATIVELY (no remote modeling file ships with the repo).
  # --moe-backend marlin is mandatory; flashinfer_cutlass REJECTS W4A16 NVFP4 and crashes
  # engine-core init on this box. --kv-cache-dtype fp8 keeps KV tiny (peaked <4% of cache).
  # gpu-mem-util 0.85: the llm.manek.sg gateway was taken DOWN for the run, so the full
  # 121 GB was free; 0.85*121 ≈ 103 GB reservation holds the 75 GB weights + KV cleanly.
  docker run --gpus all --ipc=host -p 8000:8000 \
    -v ~/Desktop/reap-nvfp8/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4A16:/model:ro \
    vllm/vllm-openai:nightly-aarch64 \
    /model \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --kv-cache-dtype fp8 \
    --quantization compressed-tensors --moe-backend marlin --trust-remote-code \
    --served-model-name minimax-m25-reap-nvfp4
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model minimax-m25-reap-nvfp4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Done — self-quantized NVFP4A16, serves on vLLM/marlin.** This is the **REAP-pruned** MiniMax-M2
(Cerebras `MiniMax-M2.5-REAP-139B-A10B`: 139B total / 10B active, 154 experts (pruned from 256),
62 layers, `MiniMaxM2ForCausalLM`, a reasoning/agentic model). REAP is router-weighted
expert-activation pruning, so this is a *second* compression stage stacked under our NVFP4. No
published NVFP4 of this REAP checkpoint exists, so we published ours →
**[`gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4`](https://huggingface.co/gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4)**.
Unlike the GLM-Air REAP, **no BF16/FP16 source exists** — the only source is **block-FP8**
(`float8_e4m3fn` + `weight_scale_inv` [128,128]), 131 GB / 27 shards, which the streaming quantizer
dequantizes to BF16 on the fly per shard (`w_bf16 = w_fp8 × scale_inv`) before computing NVFP4.

**Result (conc-32 ShareGPT, 65536 ctx):** prefill **127.93 tok/s**, decode **120.0 tok/s**,
ttft_median 866 ms, tpot_median 257 ms, **0 errors** (465/1000 prompts in the 900 s cap). On-disk
checkpoint **75 GB**; runtime footprint ~109 GB = vLLM 0.85 reservation (≈75 GB weights +
KV/activations; KV peaked <4% of cache with fp8 KV). Slower than the smaller GLM-Air REAP (decode
158 tok/s) despite fewer active params (10B vs 12B) — the larger total weight mass (75 vs 51 GB) and
154 vs 96 experts cost more memory-bandwidth per decoded token, and MiniMax emits long reasoning
traces.

**How it was quantized — `~/Desktop/reap-nvfp8/llmc/streaming_quantize.py`:** shard-by-shard, never
loads the full model (peak RAM ≈ one shard ~5 GB). For each Linear it dequantizes block-FP8→BF16,
then reuses compressed_tensors' own primitives (`generate_gparam` fp32 global scale;
`compute_dynamic_scales_and_zp` per-block-16 fp8_e4m3 scale; `pack_fp4_to_uint8`) to emit
byte-compatible `nvfp4-pack-quantized`. **Critical correctness fix (same as GLM):** vLLM FUSES
parallel projections (q/k/v→qkv, gate/up→gate_up, expert w1/w3) and forces ONE `weight_global_scale`
per fused group; a per-tensor global scale → coherent-looking GARBAGE. Fixed with a 2-pass run:
pass 1 collects per-group min/max, pass 2 quantizes each fused group against a shared global scale.
Must set `TORCH_COMPILE_DISABLE=1` (this box's gcc can't build triton/inductor's CUDA util). Quant
ran in 453 s (28,892 weights quantized, 375 kept dense: lm_head + MoE `.gate` routers + norms);
no MTP layers present in this REAP checkpoint.

**Serve recipe (GB10):** `--quantization compressed-tensors --moe-backend marlin --trust-remote-code`
+ `--kv-cache-dtype fp8`. marlin MoE is **mandatory** — flashinfer_cutlass rejects W4A16 NVFP4 and
crashes engine-core init. `MiniMaxM2ForCausalLM` resolves natively in `nightly-aarch64` (the repo
ships `configuration_minimax_m2.py` but no `modeling_*` file — vLLM's built-in `minimax_m2` is used).
