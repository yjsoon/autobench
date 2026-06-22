---
title: Gemma 4 26B-A4B · vLLM · BF16
model: google/gemma-4-26B-A4B-it
company: Google
family: Gemma
params: 26B / 4B (MoE)
engine: vLLM
quant: BF16
quant_rationale: Stub asked for QAT-W4A16, but no W4A16/FP8 variant is published — only Google's BF16 base repo exists. Ran BF16 (official weights) and documented the deviation. See Notes.
source_repo: google/gemma-4-26B-A4B-it
download_url: https://huggingface.co/google/gemma-4-26B-A4B-it
context: 65536
modalities: [text, image]
mm_served: false
tags: [Google, Gemma, BF16, 16-40B]

status: done
prefill_toks: 212.74
decode_toks: 190.09
mem_gb: 109.20
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 08:10 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). BF16 base (no W4A16 published).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly google/gemma-4-26B-A4B-it \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model google/gemma-4-26B-A4B-it \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**Gemma's 26B MoE — runs clean, but the BF16-only weights make it the slowest "small-active" MoE
here.** Google's Gemma-4-26B-A4B (26B total / 4B active), served BF16 on vLLM.

- **Quant deviation (documented):** the stub planned **QAT-W4A16**, but Google publishes **no** W4A16 or
  FP8 variant of this model — `google/gemma-4-26B-A4B-it-qat-w4a16-ct`, `…-FP8`, and a RedHatAI FP8 all
  **404**. Only the BF16 base repo exists, so this is a **BF16** run (official weights, legit) rather
  than the intended 4-bit. The comparison below accounts for that.
- **Workload:** ShareGPT V3, concurrency 32. **745/1000, 0 errors** — clean, but **hit the 15-min cap**.
  Loaded in **810 s** (the ~52 GB BF16 weights had to download first; not a compute cost).
- **Throughput (aggregate, conc 32):** prefill **212.7 tok/s**, decode **190.1 tok/s**. TTFT median
  **539 ms**, TPOT median **164 ms** (≈6 tok/s/stream).
- **The BF16 tax is the story.** A 4B-active MoE "should" sit with the fast sparse models, yet decode
  (190) lands near the *dense* mid-size models and well under the FP8 3B-active MoEs
  (Qwen3-30B-A3B 331, Qwen3-Coder 296). The cause is precision, not sparsity: **BF16 moves 2× the bytes
  per weight that FP8 does**, and decode on GB10 is memory-bandwidth-bound, so the BF16 MoE pays roughly
  double the per-token weight traffic. Gemma's wide attention and the slightly higher 4B (vs 3B) active
  count add to it. Had an FP8/NVFP4 build been available, this would likely jump into the 280–330 band —
  a concrete illustration of how much quant format alone moves decode throughput at this scale.
- **Memory: 109.2 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (BF16 weights ≈ 52 GB — itself large, and a reason an FP8 build would help fit + speed).
- Text path benchmarked; image input not served here (`mm_served: false`).
