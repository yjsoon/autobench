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
concurrency: 4
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-4, strix-halo-vulkan]
status: done
prefill_toks: 49.38
decode_toks: 24.82
mem_gb: 9.75
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Not applicable on a base page.
measured_on: 2026-09-01
completed_at: 2026-09-01 23:14 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-q4kv-c4 llama q4_k_m base 65536 4 1000 300 256 q4kv
---

## Notes

- Headline: 24.82 aggregate decode tok/s, 49.38 aggregate prefill tok/s, 36 completed, 0 errors, 331.9 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 3663.2/8981.0/10818.9 ms; TPOT median/p95 = 122.9/173.7 ms; goodput = 0.0%.
- Peak shared-memory delta 9.75 GiB from baseline 24707772 kB to minimum 14483004 kB; VRAM cross-check peak None GiB and delta None GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- External recipe source: https://github.com/sudoingX/qwen38-mtp@431bf8a821d49e6ac919febc7a91fb2434318110; file SHA256s: README.md=47dc7f471af4668159359c6f6b14a04c9dc18fd6aa7f1ab223b0efdca110b44; serve_mtp.sh=b2e19c0b0f0b10b8aea9c8601dcc9adeb5dbbe443015895fb5ba698bd7153223; probe.py=d9ea0c9fb10435937c774d4b182a6b5ac8481837e8b3d1aa23738eeb178dfd5f.
- Recipe adaptation: applied --cache-type-k q4_0 --cache-type-v q4_0 -fa 1; the repository's probe.py was not used as the headline harness, because autobench's closed-loop ShareGPT generator supplies the recorded aggregate metrics.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c4, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 15.334 GiB; KV 1.125 GiB; recurrent state 0.592 GiB; total 17.051 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x (18 bytes / 32 values = 0.5625 bytes/value) for q4_0 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.04.712.527 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 16384, kv_unified = 'false'
  - 0.04.725.051 I srv  llama_server: model loaded
  - 0.04.725.058 I srv  llama_server: listening on http://0.0.0.0:8081
  - 4.10.740.971 W srv        update:  - cache size limit reached, removing oldest entry (size = 301.980 MiB)
  - 4.18.940.008 W srv        update:  - cache size limit reached, removing oldest entry (size = 305.042 MiB)
  - 4.31.641.804 W srv        update:  - cache size limit reached, removing oldest entry (size = 305.658 MiB)
  - 4.40.812.671 W srv        update:  - cache size limit reached, removing oldest entry (size = 304.549 MiB)
  - 5.00.289.958 W srv        update:  - cache size limit reached, removing oldest entry (size = 301.787 MiB)
  - 5.08.672.705 W srv        update:  - cache size limit reached, removing oldest entry (size = 302.191 MiB)
  - 0.05.376.753 I srv    load_model: initializing, n_slots = 4, n_ctx_slot = 16384, kv_unified = 'false'
  - 0.05.389.327 I srv  llama_server: model loaded
  - 0.05.389.332 I srv  llama_server: listening on http://0.0.0.0:8081
  - 4.11.127.464 W srv        update:  - cache size limit reached, removing oldest entry (size = 301.980 MiB)
  - 4.20.512.483 W srv        update:  - cache size limit reached, removing oldest entry (size = 304.549 MiB)
  - 4.31.853.436 W srv        update:  - cache size limit reached, removing oldest entry (size = 305.658 MiB)
  - 4.42.461.332 W srv        update:  - cache size limit reached, removing oldest entry (size = 305.042 MiB)
  - 5.00.679.774 W srv        update:  - cache size limit reached, removing oldest entry (size = 301.787 MiB)
  - 5.10.331.294 W srv        update:  - cache size limit reached, removing oldest entry (size = 302.227 MiB)
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median None/None%, temperature peak None C, power median/peak None/None W, sustained clock median None MHz.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
