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
concurrency: 8
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-8, strix-halo-vulkan]
status: done
prefill_toks: 57.15
decode_toks: 29.77
mem_gb: 10.37
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Not applicable on a base page.
measured_on: 2026-08-28
completed_at: 2026-08-28 15:58 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-c8 llama q4_k_m base 65536 8 1000 300 256
---

## Notes

- Headline: 29.77 aggregate decode tok/s, 57.15 aggregate prefill tok/s, 43 completed, 0 errors, 333.2 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 9258.4/13498.9/16355.0 ms; TPOT median/p95 = 205.1/314.45 ms; goodput = 0.0%.
- Peak shared-memory delta 10.37 GiB from baseline 22048448 kB to minimum 11175156 kB; VRAM cross-check peak 22.7 GiB and delta 20.05 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c8, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 15.334 GiB; KV 4.000 GiB; recurrent state 1.184 GiB; total 20.517 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.06.190.861 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 8192, kv_unified = 'false'
  - 0.06.208.781 I srv  llama_server: model loaded
  - 0.06.208.785 I srv  llama_server: listening on http://0.0.0.0:8081
  - 3.49.347.117 W srv        update:  - cache size limit reached, removing oldest entry (size = 304.629 MiB)
  - 4.01.610.803 W srv        update:  - cache size limit reached, removing oldest entry (size = 308.756 MiB)
  - 4.15.616.031 W srv        update:  - cache size limit reached, removing oldest entry (size = 309.693 MiB)
  - 4.15.644.271 W srv        update:  - cache size limit reached, removing oldest entry (size = 334.889 MiB)
  - 4.28.538.699 W srv        update:  - cache size limit reached, removing oldest entry (size = 316.946 MiB)
  - 4.34.189.215 W srv        update:  - cache size limit reached, removing oldest entry (size = 322.010 MiB)
  - 4.55.830.708 W srv        update:  - cache size limit reached, removing oldest entry (size = 319.821 MiB)
  - 4.57.297.911 W srv        update:  - cache size limit reached, removing oldest entry (size = 318.071 MiB)
  - 5.01.414.497 W srv        update:  - cache size limit reached, removing oldest entry (size = 313.695 MiB)
  - 5.01.424.819 W srv        update:  - cache size limit reached, removing oldest entry (size = 316.258 MiB)
  - 5.07.918.081 W srv        update:  - cache size limit reached, removing oldest entry (size = 319.071 MiB)
  - 5.13.616.806 W srv        update:  - cache size limit reached, removing oldest entry (size = 316.570 MiB)
  - 0.06.266.702 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 8192, kv_unified = 'false'
  - 0.06.279.608 I srv  llama_server: model loaded
  - 0.06.279.613 I srv  llama_server: listening on http://0.0.0.0:8081
  - 3.48.666.833 W srv        update:  - cache size limit reached, removing oldest entry (size = 304.629 MiB)
  - 3.57.976.961 W srv        update:  - cache size limit reached, removing oldest entry (size = 308.756 MiB)
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 100.0/94.0%, temperature peak 95.0 C, power median/peak 120.1/139.157 W, sustained clock median None MHz.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
