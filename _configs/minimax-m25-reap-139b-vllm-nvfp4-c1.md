---
title: MiniMax-M2.5-REAP 139B · vLLM · NVFP4 (W4A16) · conc 1 · 192K
model: cerebras/MiniMax-M2.5-REAP-139B-A10B (NVFP4 W4A16, self-quantized)
company: Cerebras (REAP) · MiniMax (base)
family: MiniMax
params: 139B / 10B (MoE, REAP-pruned from 230B/10B — 256→154 experts)
engine: vLLM
quant: NVFP4 (W4A16)
quant_rationale: Same self-quantized compressed-tensors NVFP4A16 checkpoint as the conc-32 sibling (see [base config]). This page is the single-stream serving operating point — full 192K context, conc-1, util 0.90, with tool-call + reasoning parsers enabled — i.e. how the model is actually deployed as the interactive gateway, not the throughput benchmark.
source_repo: cerebras/MiniMax-M2.5-REAP-139B-A10B
download_url: https://huggingface.co/gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4
hf_repo: gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4
hf_url: https://huggingface.co/gauravmm/MiniMax-M2.5-REAP-139B-A10B-NVFP4
context: 196608
modalities: [text]
mm_served: true
concurrency: 1
tags: [minimax-m25-reap-139b, Cerebras, MiniMax, NVFP4, 131-260B, Spark recipe, REAP, Self-Quantized, conc-1]
status: done
prefill_toks: 41.8
decode_toks: 26.99
mem_gb: 121
mem_source: system MemAvailable while serving (used ~121/121 GB, 6.7 GB free) — vLLM 0.90 reservation at 192K ctx
measured_on: 2026-06-27
completed_at: 2026-06-27 15:22 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # Interactive single-stream serving point (the live :4000 gateway), NOT the throughput run.
  # Differs from the conc-32 benchmark: max-model-len 196608 (full ctx), max-num-seqs 1,
  # gpu-memory-utilization 0.90, + minimax_m2 reasoning/tool-call parsers for agentic use.
  docker run -d --gpus all --ipc=host -p 4000:8000 --name serving-minimax-m25-reap \
    -v ~/Desktop/reap-nvfp8/models/MiniMax-M2.5-REAP-139B-A10B-NVFP4A16:/model:ro \
    vllm/vllm-openai:nightly-aarch64 \
    /model \
    --host 0.0.0.0 --port 8000 --max-model-len 196608 \
    --gpu-memory-utilization 0.90 --max-num-seqs 1 \
    --kv-cache-dtype fp8 \
    --quantization compressed-tensors --moe-backend marlin --trust-remote-code \
    --reasoning-parser minimax_m2 --enable-auto-tool-choice --tool-call-parser minimax_m2 \
    --served-model-name minimax-m25-reap-nvfp4
  python3 scripts/bench-serving.py --base-url http://localhost:4000 \
    --model minimax-m25-reap-nvfp4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 500 --max-seconds 300 --concurrency 1 --max-tokens 256
---

**The live interactive gateway point — full 192K context, single stream.** Same self-quantized NVFP4A16
checkpoint as the conc-32 throughput run ([base config]), but measured as it is actually *deployed*: one
sequence at a time, the model's full 196,608-token context, `--gpu-memory-utilization 0.90`, and the
`minimax_m2` reasoning + tool-call parsers wired up for agentic use. This is the container that stays up
on `:4000`.

**Result (conc-1 ShareGPT, 192K ctx):** prefill **41.8 tok/s**, decode **26.99 tok/s**, **34/500 prompts**
in the 300 s cap (`hit_time_cap=true`), **0 errors**. ttft_median 9.25 s and tpot_median 0.0 are
buffered-reasoning artifacts of the `minimax_m2` parser (the whole reasoning trace is held then flushed),
so the aggregate tok/s are the valid headline, not the per-token medians. Single-stream decode (27 tok/s)
is roughly a quarter of the conc-32 aggregate (120 tok/s) — expected: at conc-1 there is no batching to
amortise the 75 GB weight read per decoded token, and MiniMax emits long reasoning traces before answering.

**Memory: ~121 GB used (6.7 GB free of 127.6 GB).** Tighter than the conc-32 run's 109 GB because util is
0.90 (vs 0.85) and the KV reservation is sized for the full 192K context. It fits, but with little
headroom — this is the practical ceiling for this checkpoint on the 121 GB box.

**Qualitative behaviour (observed in interactive use):**
- **Strong on code, weak off it.** The model is a capable coding/agentic assistant but **performs poorly
  outside coding tasks** — general-knowledge, prose, and non-code reasoning are noticeably weaker. The
  double compression (REAP expert-pruning under NVFP4) plus a coding-skewed base likely concentrates what
  survived on code.
- **Occasionally misspells variable names.** In generated code it will sometimes emit a slightly-wrong
  identifier (a transposed or dropped character vs. the name it declared earlier) — a quantization/pruning
  fidelity artifact to watch for when using its output.

**Pair:** throughput sibling at [base config] (conc-32, 65536 ctx). Same `engine_image`, same checkpoint,
same marlin serve recipe — only the operating point (context / concurrency / util / parsers) differs.
