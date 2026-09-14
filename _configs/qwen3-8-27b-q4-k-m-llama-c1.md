---
title: Qwen3.8-27B · llama.cpp · Q4_K_M
model: Qwen/Qwen3.8-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: llama.cpp
speculative: none
quant: Q4_K_M
quant_rationale: >-
  GGUF UD-Q4_K_M is the selected memory-saving llama.cpp format for this UMA Vulkan machine. The inventory proved BF16 in native torch, not native low-bit arithmetic; the page therefore records the Vulkan quant path without claiming a hardware-native format.
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-1, strix-halo-vulkan]
status: done
prefill_toks: 2.99
decode_toks: 11.56
mem_gb: 5.57
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Not applicable on a base page.
measured_on: 2026-08-28
completed_at: 2026-08-28 15:23 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-c1 llama q4_k_m base 65536 1 1000 300 256
---

## Notes

- Headline: 11.56 aggregate decode tok/s, 2.99 aggregate prefill tok/s, 16 completed, 0 errors, 314.9 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 1849.0/2303.4/2643.5 ms; TPOT median/p95 = 78.7/78.9 ms; goodput = 0.0%.
- Peak shared-memory delta 5.57 GiB from baseline 21057300 kB to minimum 15217616 kB; VRAM cross-check peak 21.66 GiB and delta 19.01 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 15.334 GiB; KV 4.000 GiB; recurrent state 0.148 GiB; total 19.482 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.06.025.550 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.06.038.038 I srv  llama_server: model loaded
  - 0.06.038.042 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.05.107.221 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.05.120.009 I srv  llama_server: model loaded
  - 0.05.120.013 I srv  llama_server: listening on http://0.0.0.0:8081
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 99.0/98.0%, temperature peak 94.0 C, power median/peak 120.15350000000001/135.033 W, sustained clock median None MHz.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
