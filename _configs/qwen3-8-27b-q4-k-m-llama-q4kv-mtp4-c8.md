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
prefill_toks: 40.22
decode_toks: 21.88
mem_gb: 10.38
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 52.23% (10317 accepted / 19754 generated); weighted mean acceptance length 3.08; num_speculative_tokens=4.
measured_on: 2026-09-01
completed_at: 2026-09-01 22:49 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c8 llama q4_k_m mtp4 65536 8 1000 300 256 q4kv
---

## Notes

- Headline: 21.88 aggregate decode tok/s, 40.22 aggregate prefill tok/s, 34 completed, 0 errors, 350.0 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 5318.6/16167.7/18482.9 ms; TPOT median/p95 = 311.85/447.35 ms; goodput = 0.0%.
- Peak shared-memory delta 10.38 GiB from baseline 24612696 kB to minimum 13723968 kB; VRAM cross-check peak None GiB and delta None GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- External recipe source: https://github.com/sudoingX/qwen38-mtp@431bf8a821d49e6ac919febc7a91fb2434318110; file SHA256s: README.md=47dc7f471af4668159359c6f6b14a04c9dc18fd6aa7f1ab223b0efdca110b44; serve_mtp.sh=b2e19c0b0f0b10b8aea9c8601dcc9adeb5dbbe443015895fb5ba698bd7153223; probe.py=d9ea0c9fb10435937c774d4b182a6b5ac8481837e8b3d1aa23738eeb178dfd5f.
- Recipe adaptation: applied --cache-type-k q4_0 --cache-type-v q4_0 -fa 1 --spec-type draft-mtp --spec-draft-n-max 4; the repository's probe.py was not used as the headline harness, because autobench's closed-loop ShareGPT generator supplies the recorded aggregate metrics.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c8, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 1.125 GiB; recurrent state 1.184 GiB; total 18.918 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x (18 bytes / 32 values = 0.5625 bytes/value) for q4_0 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.06.555.986 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 8192, kv_unified = 'false'
  - 0.06.653.946 I srv  llama_server: model loaded
  - 0.06.653.951 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.13.220.457 I slot print_timing: id  7 | task 0 | draft acceptance = 0.56579 (   43 accepted /    76 generated), mean len =  3.26
  - 0.43.808.436 I slot print_timing: id  5 | task 25 | draft acceptance = 0.48000 (   48 accepted /   100 generated), mean len =  2.92
  - 1.00.642.139 I slot print_timing: id  7 | task 23 | draft acceptance = 0.48295 (   85 accepted /   176 generated), mean len =  2.93
  - 1.06.051.485 I slot print_timing: id  2 | task 28 | draft acceptance = 0.54082 (  106 accepted /   196 generated), mean len =  3.16
  - 1.19.268.782 I slot print_timing: id  6 | task 24 | draft acceptance = 0.77108 (  192 accepted /   249 generated), mean len =  4.05
  - 1.34.413.322 I slot print_timing: id  0 | task 30 | draft acceptance = 0.55911 (  175 accepted /   313 generated), mean len =  3.22
  - 1.42.270.941 I slot print_timing: id  4 | task 27 | draft acceptance = 0.47443 (  167 accepted /   352 generated), mean len =  2.90
  - 1.44.715.239 I slot print_timing: id  3 | task 26 | draft acceptance = 0.45179 (  164 accepted /   363 generated), mean len =  2.80
  - 1.54.189.991 I slot print_timing: id  1 | task 29 | draft acceptance = 0.39695 (  156 accepted /   393 generated), mean len =  2.58
  - 2.07.839.426 I slot print_timing: id  5 | task 59 | draft acceptance = 0.50000 (  169 accepted /   338 generated), mean len =  2.99
  - 2.18.206.549 I slot print_timing: id  6 | task 100 | draft acceptance = 0.84483 (  196 accepted /   232 generated), mean len =  4.38
  - 2.18.220.023 I slot print_timing: id  7 | task 79 | draft acceptance = 0.57097 (  177 accepted /   310 generated), mean len =  3.27
  - 2.30.700.962 I slot print_timing: id  2 | task 85 | draft acceptance = 0.50445 (  170 accepted /   337 generated), mean len =  3.00
  - 2.41.911.152 I slot print_timing: id  0 | task 118 | draft acceptance = 0.71863 (  189 accepted /   263 generated), mean len =  3.86
  - 2.53.847.498 I slot print_timing: id  1 | task 140 | draft acceptance = 0.82278 (  195 accepted /   237 generated), mean len =  4.25
  - 2.53.854.114 I slot print_timing: id  4 | task 127 | draft acceptance = 0.65018 (  184 accepted /   283 generated), mean len =  3.59
  - 2.57.095.668 I slot print_timing: id  3 | task 130 | draft acceptance = 0.65248 (  184 accepted /   282 generated), mean len =  3.59
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median None/None%, temperature peak None C, power median/peak None/None W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 52.23% (10317 accepted / 19754 generated); weighted mean acceptance length 3.08; num_speculative_tokens=4.
- Cross-check anchor: measured MTP acceptance 52.23%; this is not consistent with the previous-generation ShareGPT MTP anchor of 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Anchor investigation: the gap is recorded because this is a newer Qwen3.8 checkpoint and the qwen38-mtp run uses draft depth 4; server logs emitted draft-acceptance lines, so no silent-disabled fallback was observed. The cause of the lower acceptance was not isolated.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
