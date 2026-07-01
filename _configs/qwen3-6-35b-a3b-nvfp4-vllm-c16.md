---
title: Qwen3.6-35B-A3B · vLLM · NVFP4 · conc 16
model: Qwen/Qwen3.6-35B-A3B
company: Alibaba
family: Qwen
params: 35B / 3B (MoE)
engine: vLLM
quant: NVFP4
quant_rationale: conc-16 base (non-spec) point of the Qwen3.6-35B-A3B NVFP4 sweep — matched no-spec baseline for the MTP/DFlash conc-16 rows (EXPERIMENTS.md #4/#14). NVIDIA official ModelOpt NVFP4, same recipe as the published conc-32 base; only --max-num-seqs differs. Completes the base line of the speedup-decay curve.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 16
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-16]
status: done
prefill_toks: 349.96
decode_toks: 332.42
mem_gb: 108.83
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85)
measured_on: 2026-07-01
completed_at: 2026-07-01 14:58 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:68e23ddd982ad5642e21354c2242a3a86d31a3ea83f5937e5c3867942dc6595b
run_command: |
  # conc-16 base (no spec). Same recipe as the published conc-32 base — only --max-num-seqs differs.
  scripts/bench-vllm-serving.sh nvidia/Qwen3.6-35B-A3B-NVFP4 65536 16 1000 600 256 \
    --quantization modelopt --trust-remote-code --reasoning-parser qwen3
  # 784/1000 prompts (hit 600 s cap), 0 errors. ready after ~385 s.
---

**conc-16 base (no-spec) point of the Qwen3.6-35B-A3B NVFP4 sweep** — completes the matched base line for
the money-chart concurrencies (EXPERIMENTS.md #4/#14). Same NVIDIA ModelOpt NVFP4 recipe as the published
conc-32 base; only `--max-num-seqs` changes.

- **Result (conc 16):** prefill 349.96 / decode **332.42** tok/s aggregate; 784/1000 prompts (hit the 600 s
  cap), **0 errors**; peak mem 108.8 GB.
- **The base line is clean and monotone:** decode 74.7 (c1) → 113.2 (c2) → 173.8 (c4) → 241.8 (c8) → **332.4
  (c16)** → 430.8 (c32). A textbook throughput-vs-concurrency curve, no anomalies.
- **MTP-vs-base ratio is non-monotone — but that's an MTP-measurement-window artifact, not a real curve
  shape.** Ratios: +25.6% (c1) / +42.4% (c2) / +33.8% (c4) / +19.5% (c8) / **+30.3% (c16)** / +25.7% (c32).
  The base points here are all one clean session (600 s caps); the MTP points mix dates/caps — c1/c8/c32 are
  the 2026-06-23 runs (c8 used a 300 s / 500-prompt cap), c2/c4/c16 are today. The two **cleanest
  same-session, same-600 s-cap** comparisons are **c4 (+33.8%) and c16 (+30.3%)**, bracketing ~**+30%**. So
  the honest takeaway is **MTP wins a robust ~+20–42% across the whole sweep**, centered ~+30%, without a
  clean monotone decay. For the single-session apples-to-apples decay story, use the DFlash-vs-MTP money
  chart (all one session) rather than this base-vs-MTP ratio.
- **To fully clean the #14 curve** would require re-running MTP c1/c8 at the 600 s cap to match; logged as a
  follow-up rather than smoothed over here.
- TPOT 0.0 = `qwen3` reasoning-parser client artifact — decode tok/s is the reported metric.
- Sweep siblings: [`-c1`](qwen3-6-35b-a3b-nvfp4-vllm-c1) · [`-c2`](qwen3-6-35b-a3b-nvfp4-vllm-c2) ·
  [`-c4`](qwen3-6-35b-a3b-nvfp4-vllm-c4) · [`-c8`](qwen3-6-35b-a3b-nvfp4-vllm-c8) ·
  [`c32` (main)](qwen3-6-35b-a3b-nvfp4-vllm). MTP counterpart: [`-mtp-c16`](qwen3-6-35b-a3b-nvfp4-vllm-mtp-c16).
