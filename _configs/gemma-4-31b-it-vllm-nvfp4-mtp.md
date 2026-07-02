---
title: Gemma 4 31B · vLLM · NVFP4 + MTP
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 31B (dense)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + Google's official MTP assistant drafter (google/gemma-4-31B-it-assistant) via vLLM's native gemma-4 MTP path. The MTP counterpart to the done NVFP4+EAGLE3 row — completes the base/MTP/EAGLE3 comparison for the heaviest Gemma-4 dense model.
source_repo: nvidia/Gemma-4-31B-IT-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 375.5
decode_toks: 323.5
mem_gb: 108.44
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + Gemma4 MTP head
spec_acceptance: 58-65% avg draft acceptance · mean acceptance length ~2.8 (2.73-2.95) · per-position ~0.79/0.60/0.44 (num_speculative_tokens=3)
measured_on: 2026-07-02
completed_at: 2026-07-02 21:39 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:e414712fdc04f61d98ccc58cb61232a0587a8c024544e9e6cf12f97b19b38172
run_command: |
  # UNBLOCKED (2026-07-02) on the current nightly-aarch64 — same as the 26B-A4B MTP row. The
  # "NVFP4 <-> gemma4_assistant mutually exclusive across images" wall was over-generalized from the
  # E4B elastic checkpoint; this non-elastic dense NVFP4 checkpoint resolves both
  # Gemma4ForConditionalGeneration (modelopt_fp4) and Gemma4MTPModel and serves cleanly. No custom image.
  VLLM_IMAGE=vllm/vllm-openai:nightly-aarch64 scripts/bench-vllm-serving.sh nvidia/Gemma-4-31B-IT-NVFP4 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-31B-it-assistant","num_speculative_tokens":3}'
  # 1000/1000 prompts, 0 errors, 717.7 s. TTFT median 685.1 ms, TPOT median 86.3 ms, req thr 1.393/s.
  # Ready after 707 s. SpecDecoding: mean accept-len ~2.8, avg draft acceptance ~58-65%.
---

**DONE — MTP beats EAGLE3 on the heaviest Gemma-4 too.** NVIDIA NVFP4 base + Google's official
`google/gemma-4-31B-it-assistant` MTP drafter, on the maintained vLLM `nightly-aarch64`. Decode
**323.5 tok/s** — the fastest 31B config, ahead of both base and NVFP4+EAGLE3.

- **Result (conc 32):** prefill **375.5** tok/s, decode **323.5** tok/s aggregate; **1000/1000, 0
  errors** in **717.7 s**. TTFT median **685.1 ms**, TPOT median **86.3 ms**, req throughput **1.393/s**.
  Peak mem **108.44 GB** (vLLM static KV reservation at util 0.85 + MTP head). Ready after 707 s.
- **base vs +MTP vs +EAGLE3 (same model/engine/quant, conc 32):**

  | Config | decode | TPOT med | vs base | vs EAGLE3 |
  |---|--:|--:|--:|--:|
  | NVFP4 (base) | 166.96 | 176 ms | — | — |
  | NVFP4 + EAGLE3 | 264.72 | 108.8 ms | +58.6% | — |
  | **NVFP4 + MTP** | **323.5** | **86.3 ms** | **+93.8%** | **+22.2%** |

  **MTP wins again** — +94% over base and **+22% over EAGLE3**. Same verdict as the 26B-A4B row: on the
  heavier Gemma-4 models the draft-efficient native MTP path outruns the EAGLE3 head.
- **Acceptance:** mean accept-len **~2.8** (windowed 2.73-2.95), avg draft acceptance **~58-65%**,
  per-position **~0.79 / 0.60 / 0.44** at `num_speculative_tokens=3` — the *best* acceptance of the
  Gemma-4 MTP runs (dense 31B target + its matched assistant drafter), and comfortably clear of the
  near-1.0 that would signal a target/draft mismatch.
- **No image wall.** Loads on stock `nightly-aarch64` (resolved `Gemma4ForConditionalGeneration`,
  `quantization=modelopt_fp4`, `FlashInferCutlassNvFp4`, TRITON_ATTN forced for heterogeneous heads) with
  the MTP drafter attached, 0 errors — see the 26B-A4B MTP page and `notes/INCOMPATIBILITIES.md`.
- Text path benchmarked (`mm_served: false`).
