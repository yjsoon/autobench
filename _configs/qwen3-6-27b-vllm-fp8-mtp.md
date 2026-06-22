---
title: Qwen3.6-27B · vLLM · FP8 + MTP
model: Qwen/Qwen3.6-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: MTP
quant: FP8
quant_rationale: Qwen3.6-27B FP8 + the model's own native MTP module (mtp.safetensors ships in the base repo) — built-in multi-token-prediction speculative decoding, no separate draft.
source_repo: Qwen/Qwen3.6-27B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.6-27B-FP8
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [qwen3.6-27b, Alibaba, Qwen, FP8, 16-40B, conc-32]
status: done
prefill_toks: 241.68
decode_toks: 240.92
mem_gb: 107.09
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + MTP head
spec_acceptance: 67% avg draft acceptance · mean acceptance length 3.0 · per-position 0.84/0.67/0.51
measured_on: 2026-06-22
completed_at: 2026-06-22 18:46 +08
run_command: |
  # base Qwen3.6-27B-FP8 + native MTP (mtp.safetensors ships in-repo) via vLLM --speculative-config.
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly Qwen/Qwen3.6-27B-FP8 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 --gpu-memory-utilization 0.85 --max-num-seqs 32 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  python3 scripts/bench-serving.py --base-url http://localhost:8000 --model Qwen/Qwen3.6-27B-FP8 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256

# SpecDecoding metrics (vLLM, steady-state): Mean acceptance length ~3.0, Avg draft acceptance ~67%,
# per-position acceptance 0.84 / 0.67 / 0.51 (num_speculative_tokens=3).
---

**Native MTP is a clean +56% win on Qwen3.6-27B — and the first config with a captured acceptance
rate, which explains exactly why.** Qwen3.6-27B FP8 + the model's built-in MTP head, on vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **867/1000, 0 errors** — clean, hit the cap. Loaded +
  CUDA-graph captured in **504 s** (MTP adds compile time).
- **Throughput (aggregate, conc 32):** prefill **241.7 tok/s**, decode **240.9 tok/s** vs the base
  Qwen3.6-27B's **154.7** → **+56% decode** from MTP at conc 32.
- **Acceptance rate (captured from vLLM's `SpecDecoding metrics`):** **mean acceptance length ≈ 3.0**,
  **avg draft acceptance ≈ 67%**, per-position **0.84 / 0.67 / 0.51** (for the 3 draft positions). With
  `num_speculative_tokens=3`, ~3 tokens are emitted per target step on average (the always-accepted
  token + ~2 accepted drafts). High acceptance on ShareGPT's fairly predictable chat continuations is
  what converts into the +56%.
- **Why this won where gpt-oss-120b's EAGLE3 lost — same conc 32, opposite size.** A **27B** model still
  has GB10 compute headroom at batch 32, so the MTP draft+verify rides on otherwise-idle FLOPs and the
  ~3× tokens/step turns into real throughput. The **120B** gpt-oss was already compute-saturated, so the
  same idea went −45%. Acceptance ~67% here is healthy; even with good acceptance, a saturated large
  model wouldn't benefit. **Headroom × acceptance is the predictor** — and this 27B has both.
- **Memory: 107.1 GB is the vLLM `--gpu-memory-utilization 0.85` reservation** (+ the small MTP head),
  not the footprint (FP8 weights ≈ 27 GB).
