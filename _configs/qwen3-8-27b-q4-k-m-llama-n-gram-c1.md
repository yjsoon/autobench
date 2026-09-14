---
title: Qwen3.8-27B · llama.cpp · Q4_K_M + n-gram
model: Qwen/Qwen3.8-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: llama.cpp
speculative: n-gram
quant: Q4_K_M
quant_rationale: >-
  GGUF UD-Q4_K_M is the selected memory-saving llama.cpp format for this UMA Vulkan machine. The inventory proved BF16 in native torch, not native low-bit arithmetic; the page therefore records the Vulkan quant path without claiming a hardware-native format.
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-1, strix-halo-vulkan]
status: done
prefill_toks: 2.96
decode_toks: 11.43
mem_gb: 6.01
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 12.86% (36 accepted / 280 generated); weighted mean acceptance length 5.85; num_speculative_tokens=n/a; llama.cpp ngram-simple-size-n=12 and ngram-simple-size-m=48.
measured_on: 2026-08-28
completed_at: 2026-08-28 16:13 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-n-gram-c1 llama q4_k_m ngram 65536 1 1000 300 256
---

## Notes

- Headline: 11.43 aggregate decode tok/s, 2.96 aggregate prefill tok/s, 16 completed, 0 errors, 318.3 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 1847.7/2290.9/2646.6 ms; TPOT median/p95 = 78.8/84.2 ms; goodput = 0.0%.
- Peak shared-memory delta 6.01 GiB from baseline 21547608 kB to minimum 15249608 kB; VRAM cross-check peak 21.68 GiB and delta 19.03 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 15.334 GiB; KV 4.000 GiB; recurrent state 0.148 GiB; total 19.482 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.06.091.097 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.06.103.885 I srv  llama_server: model loaded
  - 0.06.103.890 I srv  llama_server: listening on http://0.0.0.0:8081
  - 4.47.511.384 I slot print_timing: id  0 | task 2962 | draft acceptance = 0.10000 (    2 accepted /    20 generated), mean len =  3.00
  - 5.11.936.148 I slot print_timing: id  0 | task 3219 | draft acceptance = 0.13333 (   16 accepted /   120 generated), mean len =  6.33
  - 0.05.245.610 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.05.258.467 I srv  llama_server: model loaded
  - 0.05.258.473 I srv  llama_server: listening on http://0.0.0.0:8081
  - 4.46.949.243 I slot print_timing: id  0 | task 2962 | draft acceptance = 0.10000 (    2 accepted /    20 generated), mean len =  3.00
  - 5.11.215.479 I slot print_timing: id  0 | task 3219 | draft acceptance = 0.13333 (   16 accepted /   120 generated), mean len =  6.33
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 98.0/97.0%, temperature peak 94.0 C, power median/peak 120.131/135.753 W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 12.86% (36 accepted / 280 generated); weighted mean acceptance length 5.85; num_speculative_tokens=n/a; llama.cpp ngram-simple-size-n=12 and ngram-simple-size-m=48.
- Cross-check anchor: the 67% value is a previous-generation ShareGPT MTP anchor, not an n-gram expectation; this n-gram result is method-specific and is not directly comparable to that anchor.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
