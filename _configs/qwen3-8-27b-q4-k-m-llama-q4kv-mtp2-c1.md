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
prefill_toks: 17.50
decode_toks: 19.68
mem_gb: 9.21
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 72.80% (7018 accepted / 9640 generated); weighted mean acceptance length 2.45; num_speculative_tokens=2.
measured_on: 2026-09-01
completed_at: 2026-09-01 22:03 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1 llama q4_k_m mtp2 65536 1 1000 300 256 q4kv
---

## Notes

- Headline: 19.68 aggregate decode tok/s, 17.50 aggregate prefill tok/s, 27 completed, 0 errors, 302.2 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 1725.8/4436.1/8603.0 ms; TPOT median/p95 = 42.45/47.7 ms; goodput = 77.8%.
- Peak shared-memory delta 9.21 GiB from baseline 24675504 kB to minimum 15015620 kB; VRAM cross-check peak None GiB and delta None GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- External recipe source: https://github.com/sudoingX/qwen38-mtp@431bf8a821d49e6ac919febc7a91fb2434318110; file SHA256s: README.md=47dc7f471af4668159359c6f6b14a04c9dc18fd6aa7f1ab223b0efdca110b44; serve_mtp.sh=b2e19c0b0f0b10b8aea9c8601dcc9adeb5dbbe443015895fb5ba698bd7153223; probe.py=d9ea0c9fb10435937c774d4b182a6b5ac8481837e8b3d1aa23738eeb178dfd5f.
- Recipe adaptation: applied --cache-type-k q4_0 --cache-type-v q4_0 -fa 1 --spec-type draft-mtp --spec-draft-n-max 2; the repository's probe.py was not used as the headline harness, because autobench's closed-loop ShareGPT generator supplies the recorded aggregate metrics.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 1.125 GiB; recurrent state 0.148 GiB; total 17.882 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x (18 bytes / 32 values = 0.5625 bytes/value) for q4_0 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.05.209.698 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.05.309.368 I srv  llama_server: model loaded
  - 0.05.309.373 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.11.873.131 I slot print_timing: id  0 | task 0 | draft acceptance = 0.84783 (   39 accepted /    46 generated), mean len =  2.70
  - 0.24.865.106 I slot print_timing: id  0 | task 27 | draft acceptance = 0.67281 (  146 accepted /   217 generated), mean len =  2.34
  - 0.31.606.462 I slot print_timing: id  0 | task 138 | draft acceptance = 0.67000 (   67 accepted /   100 generated), mean len =  2.34
  - 0.42.720.370 I slot print_timing: id  0 | task 191 | draft acceptance = 0.91667 (  165 accepted /   180 generated), mean len =  2.83
  - 0.56.283.565 I slot print_timing: id  0 | task 284 | draft acceptance = 0.60173 (  139 accepted /   231 generated), mean len =  2.20
  - 1.00.955.192 I slot print_timing: id  0 | task 403 | draft acceptance = 0.68333 (   41 accepted /    60 generated), mean len =  2.37
  - 1.08.767.873 I slot print_timing: id  0 | task 436 | draft acceptance = 0.73387 (   91 accepted /   124 generated), mean len =  2.47
  - 1.21.404.167 I slot print_timing: id  0 | task 501 | draft acceptance = 0.79188 (  156 accepted /   197 generated), mean len =  2.58
  - 1.33.959.735 I slot print_timing: id  0 | task 603 | draft acceptance = 0.68692 (  147 accepted /   214 generated), mean len =  2.37
  - 1.45.232.798 I slot print_timing: id  0 | task 714 | draft acceptance = 0.66667 (  128 accepted /   192 generated), mean len =  2.33
  - 1.57.020.081 I slot print_timing: id  0 | task 813 | draft acceptance = 0.77000 (  154 accepted /   200 generated), mean len =  2.54
  - 2.09.322.730 I slot print_timing: id  0 | task 917 | draft acceptance = 0.71770 (  150 accepted /   209 generated), mean len =  2.43
  - 2.20.417.024 I slot print_timing: id  0 | task 1025 | draft acceptance = 0.85638 (  161 accepted /   188 generated), mean len =  2.71
  - 2.32.556.527 I slot print_timing: id  0 | task 1122 | draft acceptance = 0.72596 (  151 accepted /   208 generated), mean len =  2.45
  - 2.43.614.149 I slot print_timing: id  0 | task 1229 | draft acceptance = 0.87568 (  162 accepted /   185 generated), mean len =  2.74
  - 2.54.798.174 I slot print_timing: id  0 | task 1325 | draft acceptance = 0.88587 (  163 accepted /   184 generated), mean len =  2.77
  - 3.05.252.386 I slot print_timing: id  0 | task 1420 | draft acceptance = 0.97110 (  168 accepted /   173 generated), mean len =  2.93
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median None/None%, temperature peak None C, power median/peak None/None W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 72.80% (7018 accepted / 9640 generated); weighted mean acceptance length 2.45; num_speculative_tokens=2.
- Cross-check anchor: measured MTP acceptance 72.80%; this is consistent with the previous-generation ShareGPT MTP anchor of 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
