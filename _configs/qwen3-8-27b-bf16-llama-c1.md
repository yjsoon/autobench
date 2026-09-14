---
title: Qwen3.8-27B · llama.cpp · BF16
model: Qwen/Qwen3.8-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: llama.cpp
speculative: none
quant: BF16
quant_rationale: >-
  BF16 is the wide baseline and the narrowest format proven to execute by the native torch probe. Fit and serving behavior are measured rather than inferred from parameter count.
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.8-27b, Alibaba, Qwen, BF16, 16-40B, conc-1, strix-halo-vulkan]
status: done
prefill_toks: 1.79
decode_toks: 4.11
mem_gb: 3.75
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Not applicable on a base page.
measured_on: 2026-08-28
completed_at: 2026-08-28 18:08 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-bf16-llama-c1 llama bf16 base 65536 1 1000 300 256
---

## Notes

- Headline: 4.11 aggregate decode tok/s, 1.79 aggregate prefill tok/s, 7 completed, 0 errors, 338.2 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 2375.2/3924.9/4255.9 ms; TPOT median/p95 = 231.05/232.35 ms; goodput = 0.0%.
- Peak shared-memory delta 3.75 GiB from baseline 21442356 kB to minimum 17512020 kB; VRAM cross-check peak 55.52 GiB and delta 52.88 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 50.904 GiB; KV 4.000 GiB; recurrent state 0.148 GiB; total 55.052 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 30.01.580.967 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 30.01.596.485 I srv  llama_server: model loaded
  - 30.01.596.497 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.40.254.588 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.40.271.064 I srv  llama_server: model loaded
  - 0.40.271.078 I srv  llama_server: listening on http://0.0.0.0:8081
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 100.0/0.0%, temperature peak 96.0 C, power median/peak 17.87/111.521 W, sustained clock median None MHz.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
