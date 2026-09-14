---
title: Qwen3.8-27B · llama.cpp · Q4_K_M + MTP
model: Qwen/Qwen3.8-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: llama.cpp
speculative: MTP
quant: Q4_K_M
quant_rationale: >-
  GGUF UD-Q4_K_M is the selected memory-saving llama.cpp format for this UMA Vulkan machine. The inventory proved BF16 in native torch, not native low-bit arithmetic; the page therefore records the Vulkan quant path without claiming a hardware-native format.
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 32
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-32, strix-halo-vulkan]
status: blocked
prefill_toks: n/a
decode_toks: n/a
mem_gb: n/a
mem_source: >-
  Shared UMA method would use baseline host MemAvailable minus the 2-second minimum, with one accelerator summed. No publishable peak was recorded for this blocked attempt; static reservations are not a footprint.
spec_acceptance: >-
  Acceptance metric was not emitted or was not parseable before teardown; speculative result is blocked; num_speculative_tokens=3.
measured_on: 2026-08-28
completed_at: n/a
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  n/a; no runnable command was recorded
---

## Notes

- Status: blocked. Exact observed error/reason: Base Q4_K_M llama.cpp c32 was not launched because standardized c8 already exceeded the 2x median-TTFT SLO stop threshold (9258.4 ms > 4000 ms); MTP c32 has no acceptance or server error because it was not launched.
- No complete memory trace was published for this attempt.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c32, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 4.000 GiB; recurrent state 4.734 GiB; total 25.344 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- The engine emitted no parseable numeric KV/weight/memory breakdown before teardown; no value was inferred.
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median None/None%, temperature peak None C, power median/peak None/None W, sustained clock median None MHz.
- Speculation acceptance: Acceptance metric was not emitted or was not parseable before teardown; speculative result is blocked; num_speculative_tokens=3.
- Cross-check anchor: not assessable because this MTP configuration was blocked before acceptance metrics; the previous-generation ShareGPT MTP anchor is 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
- Candidate note: Base Q4_K_M llama.cpp c32 was not launched because standardized c8 already exceeded the 2x median-TTFT SLO stop threshold (9258.4 ms > 4000 ms); MTP c32 has no acceptance or server error because it was not launched.
