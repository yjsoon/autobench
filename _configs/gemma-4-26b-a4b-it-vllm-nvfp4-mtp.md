---
title: Gemma 4 26B-A4B · vLLM · NVFP4 + MTP
model: google/gemma-4-26B-A4B-it
company: Google
family: Gemma
params: 26B / 4B (MoE)
engine: vLLM
speculative: MTP (Google assistant drafter)
quant: NVFP4
quant_rationale: NVIDIA NVFP4 base (modelopt) + Google's official MTP assistant drafter (google/gemma-4-26B-A4B-it-assistant) via vLLM's native gemma-4 MTP path. The MTP counterpart to the done NVFP4+EAGLE3 row — completes the base/MTP/EAGLE3 comparison for the heavier MoE model.
source_repo: nvidia/Gemma-4-26B-A4B-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
concurrency: 32
tags: [gemma-4-26b-a4b, Google, Gemma, NVFP4, 16-40B, conc-32]
status: done
prefill_toks: 791.72
decode_toks: 692.14
mem_gb: 109.01
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85) + Gemma4 MTP head
spec_acceptance: 55-59% avg draft acceptance · mean acceptance length ~2.7 (2.29-2.85) · per-position ~0.76/0.57/0.42 (num_speculative_tokens=3)
measured_on: 2026-07-02
completed_at: 2026-07-02 21:15 +0800
engine_image: vllm/vllm-openai:nightly-aarch64@sha256:e414712fdc04f61d98ccc58cb61232a0587a8c024544e9e6cf12f97b19b38172
run_command: |
  # UNBLOCKED (2026-07-02): the "NVFP4 <-> gemma4_assistant MTP mutually exclusive across images" wall
  # was over-generalized from the E4B elastic checkpoint. The current nightly-aarch64 resolves BOTH
  # Gemma4ForConditionalGeneration (NVFP4 base, modelopt_fp4 / FlashInferCutlass) AND Gemma4MTPModel
  # (the google assistant drafter) and serves cleanly — no custom image needed. The E4B tie_weights
  # NotImplementedError does NOT fire on this non-elastic MoE checkpoint.
  VLLM_IMAGE=vllm/vllm-openai:nightly-aarch64 scripts/bench-vllm-serving.sh nvidia/Gemma-4-26B-A4B-NVFP4 65536 32 1000 900 256 \
    --speculative-config '{"method":"mtp","model":"google/gemma-4-26B-A4B-it-assistant","num_speculative_tokens":3}'
  # 1000/1000 prompts, 0 errors, 340.4 s. TTFT median 333.7 ms, TPOT median 41.8 ms, req thr 2.938/s.
  # Ready after 503 s. SpecDecoding: mean accept-len ~2.7, avg draft acceptance ~55-59%.
---

**DONE — and MTP BEATS EAGLE3 on the heavier MoE model.** NVIDIA NVFP4 base + Google's official
`google/gemma-4-26B-A4B-it-assistant` MTP drafter, on the maintained vLLM `nightly-aarch64`. This is
**the fastest 26B-A4B config measured** — decode **692.1 tok/s**, past both the NVFP4 base and the
NVFP4+EAGLE3 row.

- **Result (conc 32):** prefill **791.72** tok/s, decode **692.14** tok/s aggregate; **1000/1000, 0
  errors** in **340.4 s**. TTFT median **333.7 ms**, TPOT median **41.8 ms**, req throughput **2.938/s**.
  Peak mem **109.01 GB** (vLLM static KV reservation at util 0.85 + MTP head). Ready after 503 s.
- **base vs +MTP vs +EAGLE3 (same model/engine/quant, conc 32):**

  | Config | decode | TPOT med | vs base | vs EAGLE3 |
  |---|--:|--:|--:|--:|
  | NVFP4 (base) | 384.1 | 79.3 ms | — | — |
  | NVFP4 + EAGLE3 | 541.0 | 54.1 ms | +40.8% | — |
  | **NVFP4 + MTP** | **692.1** | **41.8 ms** | **+80.2%** | **+28.0%** |

  **MTP wins outright here** — +80% over base and **+28% over EAGLE3**, at the lowest per-token latency.
  The native Gemma-4 MTP drafter is structurally draft-efficient (drafts 3 at ~55-59% acceptance,
  accept-len ~2.7-of-3), whereas the EAGLE3 head converts fewer steps into multi-token emits on this
  MoE. So on the 26B-A4B, the **draft-efficient MTP path is the one to ship**, not EAGLE3.
- **Acceptance:** mean accept-len **~2.7** (windowed 2.29-2.85), avg draft acceptance **~55-59%**
  (brief dips to ~43%), per-position **~0.76 / 0.57 / 0.42** at `num_speculative_tokens=3` — healthy and
  well above the near-1.0 that would flag a target/draft mismatch. The NVFP4 target + BF16-trained MTP
  head are clearly matched.
- **The image wall was a mirage.** Prior notes marked this blocked by NVFP4-loading vs `gemma4_assistant`
  mutual exclusivity. That was over-generalized from the E4B *elastic* checkpoint (tied+quantized
  `lm_head` → `tie_weights NotImplementedError`). This non-elastic MoE NVFP4 checkpoint loads on
  `nightly-aarch64` (resolved `Gemma4ForConditionalGeneration`, `quantization=modelopt_fp4`,
  `FlashInferCutlassNvFp4` MoE backend, TRITON_ATTN forced for heterogeneous heads) with the MTP drafter
  attached, and serves with 0 errors. No custom image, no cu130 fallback. See
  `notes/INCOMPATIBILITIES.md` → "NVFP4+MTP DOES work on nightly-aarch64".
- Text path benchmarked (`mm_served: false`).
