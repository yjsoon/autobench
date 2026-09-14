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
concurrency: 1
tags: [qwen3.8-27b, Alibaba, Qwen, Q4_K_M, 16-40B, conc-1, strix-halo-vulkan]
status: done
prefill_toks: 26.30
decode_toks: 19.19
mem_gb: 9.58
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 60.97% (7930 accepted / 13006 generated); weighted mean acceptance length 2.82; num_speculative_tokens=3.
measured_on: 2026-08-28
completed_at: 2026-08-28 16:26 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-mtp-c1 llama q4_k_m mtp 65536 1 1000 300 256
---

## Notes

- Headline: 19.19 aggregate decode tok/s, 26.30 aggregate prefill tok/s, 28 completed, 0 errors, 321.0 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 1891.3/8477.8/11089.6 ms; TPOT median/p95 = 40.85/48.7 ms; goodput = 67.9%.
- Peak shared-memory delta 9.58 GiB from baseline 21658920 kB to minimum 11608760 kB; VRAM cross-check peak 22.49 GiB and delta 19.85 GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 4.000 GiB; recurrent state 0.148 GiB; total 20.757 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.05.696.304 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.05.744.086 I srv  llama_server: model loaded
  - 0.05.744.091 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.11.615.031 I slot print_timing: id  0 | task 0 | draft acceptance = 0.70000 (   42 accepted /    60 generated), mean len =  3.10
  - 0.24.537.301 I slot print_timing: id  0 | task 24 | draft acceptance = 0.55789 (  159 accepted /   285 generated), mean len =  2.67
  - 0.31.549.944 I slot print_timing: id  0 | task 122 | draft acceptance = 0.57246 (   79 accepted /   138 generated), mean len =  2.72
  - 0.41.863.213 I slot print_timing: id  0 | task 171 | draft acceptance = 0.84259 (  182 accepted /   216 generated), mean len =  3.53
  - 0.55.942.794 I slot print_timing: id  0 | task 247 | draft acceptance = 0.48553 (  151 accepted /   311 generated), mean len =  2.45
  - 1.00.340.582 I slot print_timing: id  0 | task 354 | draft acceptance = 0.52174 (   36 accepted /    69 generated), mean len =  2.57
  - 1.08.113.096 I slot print_timing: id  0 | task 380 | draft acceptance = 0.61728 (  100 accepted /   162 generated), mean len =  2.85
  - 1.20.408.878 I slot print_timing: id  0 | task 437 | draft acceptance = 0.68127 (  171 accepted /   251 generated), mean len =  3.04
  - 1.33.192.037 I slot print_timing: id  0 | task 524 | draft acceptance = 0.54296 (  158 accepted /   291 generated), mean len =  2.63
  - 1.43.564.924 I slot print_timing: id  0 | task 624 | draft acceptance = 0.59649 (  136 accepted /   228 generated), mean len =  2.79
  - 1.55.160.772 I slot print_timing: id  0 | task 703 | draft acceptance = 0.64368 (  168 accepted /   261 generated), mean len =  2.93
  - 2.07.743.070 I slot print_timing: id  0 | task 793 | draft acceptance = 0.56140 (  160 accepted /   285 generated), mean len =  2.68
  - 2.18.239.098 I slot print_timing: id  0 | task 891 | draft acceptance = 0.77056 (  178 accepted /   231 generated), mean len =  3.31
  - 2.29.942.977 I slot print_timing: id  0 | task 971 | draft acceptance = 0.63740 (  167 accepted /   262 generated), mean len =  2.90
  - 2.40.850.883 I slot print_timing: id  0 | task 1062 | draft acceptance = 0.73222 (  175 accepted /   239 generated), mean len =  3.19
  - 2.51.423.641 I slot print_timing: id  0 | task 1145 | draft acceptance = 0.78855 (  179 accepted /   227 generated), mean len =  3.36
  - 3.01.416.636 I slot print_timing: id  0 | task 1224 | draft acceptance = 0.83871 (  182 accepted /   217 generated), mean len =  3.49
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 100.0/95.0%, temperature peak 95.0 C, power median/peak 120.441/138.362 W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 60.97% (7930 accepted / 13006 generated); weighted mean acceptance length 2.82; num_speculative_tokens=3.
- Cross-check anchor: measured MTP acceptance 60.97%; this is consistent with the previous-generation ShareGPT MTP anchor of 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
