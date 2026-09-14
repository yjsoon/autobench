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
concurrency: 8
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-8, strix-halo-vulkan]
status: done
prefill_toks: 54.28
decode_toks: 27.57
mem_gb: 11.05
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 23.58% (249 accepted / 1056 generated); weighted mean acceptance length 10.44; num_speculative_tokens=n/a; llama.cpp ngram-simple-size-n=12 and ngram-simple-size-m=48.
measured_on: 2026-08-28
completed_at: 2026-08-28 16:39 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-n-gram-c8 llama q4_k_m ngram 65536 8 1000 300 256
---

## Notes

- Headline: 27.57 aggregate decode tok/s, 54.28 aggregate prefill tok/s, 42 completed, 0 errors, 350.3 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 7743.9/13201.9/16007.6 ms; TPOT median/p95 = 229.9/328.85 ms; goodput = 0.0%.
- Peak shared-memory delta 11.05 GiB from baseline 21873144 kB to minimum 10287284 kB; VRAM cross-check peak 22.89 GiB and delta 20.24 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c8, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 15.334 GiB; KV 4.000 GiB; recurrent state 1.184 GiB; total 20.517 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.06.277.857 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 8192, kv_unified = 'false'
  - 0.06.290.733 I srv  llama_server: model loaded
  - 0.06.290.737 I srv  llama_server: listening on http://0.0.0.0:8081
  - 2.09.513.704 I slot print_timing: id  4 | task 338 | draft acceptance = 0.13636 (   18 accepted /   132 generated), mean len =  5.50
  - 2.10.958.088 I slot print_timing: id  5 | task 335 | draft acceptance = 0.36111 (   13 accepted /    36 generated), mean len = 14.00
  - 3.34.470.684 I slot print_timing: id  2 | task 604 | draft acceptance = 0.04167 (    2 accepted /    48 generated), mean len =  3.00
  - 3.52.947.752 W srv        update:  - cache size limit reached, removing oldest entry (size = 304.629 MiB)
  - 4.02.283.644 W srv        update:  - cache size limit reached, removing oldest entry (size = 308.756 MiB)
  - 4.19.536.955 W srv        update:  - cache size limit reached, removing oldest entry (size = 309.818 MiB)
  - 4.19.562.486 W srv        update:  - cache size limit reached, removing oldest entry (size = 334.889 MiB)
  - 4.33.435.547 W srv        update:  - cache size limit reached, removing oldest entry (size = 316.946 MiB)
  - 4.41.424.583 W srv        update:  - cache size limit reached, removing oldest entry (size = 322.010 MiB)
  - 5.02.186.705 I slot print_timing: id  6 | task 887 | draft acceptance = 0.50633 (   40 accepted /    79 generated), mean len = 21.00
  - 5.03.515.428 W srv        update:  - cache size limit reached, removing oldest entry (size = 319.821 MiB)
  - 5.07.509.002 I slot print_timing: id  2 | task 868 | draft acceptance = 0.31250 (   10 accepted /    32 generated), mean len =  6.00
  - 5.08.858.801 W srv        update:  - cache size limit reached, removing oldest entry (size = 318.071 MiB)
  - 5.12.387.844 W srv        update:  - cache size limit reached, removing oldest entry (size = 313.194 MiB)
  - 5.13.836.716 W srv        update:  - cache size limit reached, removing oldest entry (size = 316.258 MiB)
  - 5.13.861.907 W srv        update:  - cache size limit reached, removing oldest entry (size = 319.071 MiB)
  - 5.24.658.503 I slot print_timing: id  4 | task 957 | draft acceptance = 0.30208 (   29 accepted /    96 generated), mean len = 15.50
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 100.0/89.0%, temperature peak 94.0 C, power median/peak 120.04/139.469 W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 23.58% (249 accepted / 1056 generated); weighted mean acceptance length 10.44; num_speculative_tokens=n/a; llama.cpp ngram-simple-size-n=12 and ngram-simple-size-m=48.
- Cross-check anchor: the 67% value is a previous-generation ShareGPT MTP anchor, not an n-gram expectation; this n-gram result is method-specific and is not directly comparable to that anchor.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
