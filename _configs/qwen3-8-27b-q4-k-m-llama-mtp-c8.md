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
concurrency: 8
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-8, strix-halo-vulkan]
status: done
prefill_toks: 34.65
decode_toks: 21.70
mem_gb: 10.42
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 60.74% (9554 accepted / 15729 generated); weighted mean acceptance length 2.82; num_speculative_tokens=3.
measured_on: 2026-08-28
completed_at: 2026-08-28 16:51 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-mtp-c8 llama q4_k_m mtp 65536 8 1000 300 256
---

## Notes

- Headline: 21.70 aggregate decode tok/s, 34.65 aggregate prefill tok/s, 33 completed, 0 errors, 342.6 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 6073.6/14288.5/16826.8 ms; TPOT median/p95 = 324.65/478.8 ms; goodput = 0.0%.
- Peak shared-memory delta 10.42 GiB from baseline 21720068 kB to minimum 10795976 kB; VRAM cross-check peak 26.59 GiB and delta 23.94 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c8, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 4.000 GiB; recurrent state 1.184 GiB; total 21.793 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.07.319.001 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 8192, kv_unified = 'false'
  - 0.07.378.961 I srv  llama_server: model loaded
  - 0.07.378.966 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.13.691.069 I slot print_timing: id  7 | task 0 | draft acceptance = 0.70000 (   42 accepted /    60 generated), mean len =  3.10
  - 0.41.902.891 I slot print_timing: id  6 | task 24 | draft acceptance = 0.52174 (   36 accepted /    69 generated), mean len =  2.57
  - 1.01.818.030 I slot print_timing: id  7 | task 25 | draft acceptance = 0.54610 (   77 accepted /   141 generated), mean len =  2.64
  - 1.08.612.688 I slot print_timing: id  5 | task 26 | draft acceptance = 0.61728 (  100 accepted /   162 generated), mean len =  2.85
  - 1.24.799.449 I slot print_timing: id  3 | task 28 | draft acceptance = 0.84259 (  182 accepted /   216 generated), mean len =  3.53
  - 1.35.041.317 I slot print_timing: id  2 | task 29 | draft acceptance = 0.68127 (  171 accepted /   251 generated), mean len =  3.04
  - 1.45.072.771 I slot print_timing: id  1 | task 30 | draft acceptance = 0.55789 (  159 accepted /   285 generated), mean len =  2.67
  - 1.47.561.347 I slot print_timing: id  4 | task 27 | draft acceptance = 0.54296 (  158 accepted /   291 generated), mean len =  2.63
  - 1.54.546.954 I slot print_timing: id  6 | task 58 | draft acceptance = 0.58442 (  135 accepted /   231 generated), mean len =  2.75
  - 1.58.328.822 I slot print_timing: id  0 | task 31 | draft acceptance = 0.47152 (  149 accepted /   316 generated), mean len =  2.41
  - 2.24.283.646 I slot print_timing: id  7 | task 83 | draft acceptance = 0.63258 (  167 accepted /   264 generated), mean len =  2.90
  - 2.36.463.995 I slot print_timing: id  5 | task 91 | draft acceptance = 0.56140 (  160 accepted /   285 generated), mean len =  2.68
  - 2.37.162.593 I slot print_timing: id  3 | task 111 | draft acceptance = 0.77056 (  178 accepted /   231 generated), mean len =  3.31
  - 2.58.833.612 I slot print_timing: id  2 | task 123 | draft acceptance = 0.62264 (  165 accepted /   265 generated), mean len =  2.85
  - 2.58.839.454 I slot print_timing: id  4 | task 138 | draft acceptance = 0.77391 (  178 accepted /   230 generated), mean len =  3.31
  - 2.59.457.730 I slot print_timing: id  1 | task 135 | draft acceptance = 0.73529 (  175 accepted /   238 generated), mean len =  3.19
  - 3.03.152.235 I slot print_timing: id  6 | task 145 | draft acceptance = 0.83871 (  182 accepted /   217 generated), mean len =  3.49
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 100.0/96.0%, temperature peak 96.0 C, power median/peak 120.1175/139.732 W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 60.74% (9554 accepted / 15729 generated); weighted mean acceptance length 2.82; num_speculative_tokens=3.
- Cross-check anchor: measured MTP acceptance 60.74%; this is consistent with the previous-generation ShareGPT MTP anchor of 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
