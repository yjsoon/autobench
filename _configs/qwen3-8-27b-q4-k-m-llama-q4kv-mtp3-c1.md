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
prefill_toks: 27.11
decode_toks: 19.89
mem_gb: 9.28
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 62.23% (8028 accepted / 12900 generated); weighted mean acceptance length 2.86; num_speculative_tokens=3.
measured_on: 2026-09-01
completed_at: 2026-09-01 22:26 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1 llama q4_k_m mtp3 65536 1 1000 300 256 q4kv
---

## Notes

- Headline: 19.89 aggregate decode tok/s, 27.11 aggregate prefill tok/s, 28 completed, 0 errors, 311.4 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 1725.6/8245.2/10852.0 ms; TPOT median/p95 = 38.65/46.95 ms; goodput = 75.0%.
- Peak shared-memory delta 9.28 GiB from baseline 24710120 kB to minimum 14976412 kB; VRAM cross-check peak None GiB and delta None GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- External recipe source: https://github.com/sudoingX/qwen38-mtp@431bf8a821d49e6ac919febc7a91fb2434318110; file SHA256s: README.md=47dc7f471af4668159359c6f6b14a04c9dc18fd6aa7f1ab223b0efdca110b44; serve_mtp.sh=b2e19c0b0f0b10b8aea9c8601dcc9adeb5dbbe443015895fb5ba698bd7153223; probe.py=d9ea0c9fb10435937c774d4b182a6b5ac8481837e8b3d1aa23738eeb178dfd5f.
- Recipe adaptation: applied --cache-type-k q4_0 --cache-type-v q4_0 -fa 1 --spec-type draft-mtp --spec-draft-n-max 3; the repository's probe.py was not used as the headline harness, because autobench's closed-loop ShareGPT generator supplies the recorded aggregate metrics.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 1.125 GiB; recurrent state 0.148 GiB; total 17.882 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x (18 bytes / 32 values = 0.5625 bytes/value) for q4_0 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.04.947.510 I srv    load_model: initializing, n_slots = 1, n_ctx_slot = 65536, kv_unified = 'false'
  - 0.05.041.252 I srv  llama_server: model loaded
  - 0.05.041.257 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.10.616.144 I slot print_timing: id  0 | task 0 | draft acceptance = 0.70000 (   42 accepted /    60 generated), mean len =  3.10
  - 0.23.498.502 I slot print_timing: id  0 | task 24 | draft acceptance = 0.57348 (  160 accepted /   279 generated), mean len =  2.70
  - 0.30.521.040 I slot print_timing: id  0 | task 121 | draft acceptance = 0.55556 (   80 accepted /   144 generated), mean len =  2.67
  - 0.40.695.414 I slot print_timing: id  0 | task 172 | draft acceptance = 0.83871 (  182 accepted /   217 generated), mean len =  3.49
  - 0.54.589.236 I slot print_timing: id  0 | task 248 | draft acceptance = 0.45370 (  147 accepted /   324 generated), mean len =  2.36
  - 1.00.902.853 I slot print_timing: id  0 | task 359 | draft acceptance = 0.68254 (   86 accepted /   126 generated), mean len =  3.05
  - 1.08.415.908 I slot print_timing: id  0 | task 404 | draft acceptance = 0.63522 (  101 accepted /   159 generated), mean len =  2.91
  - 1.20.650.936 I slot print_timing: id  0 | task 460 | draft acceptance = 0.66016 (  169 accepted /   256 generated), mean len =  2.97
  - 1.33.208.256 I slot print_timing: id  0 | task 549 | draft acceptance = 0.53952 (  157 accepted /   291 generated), mean len =  2.62
  - 1.45.526.506 I slot print_timing: id  0 | task 650 | draft acceptance = 0.55401 (  159 accepted /   287 generated), mean len =  2.66
  - 1.56.843.444 I slot print_timing: id  0 | task 749 | draft acceptance = 0.64368 (  168 accepted /   261 generated), mean len =  2.93
  - 2.08.363.978 I slot print_timing: id  0 | task 839 | draft acceptance = 0.63258 (  167 accepted /   264 generated), mean len =  2.90
  - 2.18.561.846 I slot print_timing: id  0 | task 930 | draft acceptance = 0.77056 (  178 accepted /   231 generated), mean len =  3.31
  - 2.30.206.087 I slot print_timing: id  0 | task 1010 | draft acceptance = 0.61798 (  165 accepted /   267 generated), mean len =  2.85
  - 2.40.465.780 I slot print_timing: id  0 | task 1103 | draft acceptance = 0.77729 (  178 accepted /   229 generated), mean len =  3.31
  - 2.51.601.245 I slot print_timing: id  0 | task 1183 | draft acceptance = 0.69355 (  172 accepted /   248 generated), mean len =  3.07
  - 3.00.822.219 I slot print_timing: id  0 | task 1269 | draft acceptance = 0.92118 (  187 accepted /   203 generated), mean len =  3.75
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median None/None%, temperature peak None C, power median/peak None/None W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 62.23% (8028 accepted / 12900 generated); weighted mean acceptance length 2.86; num_speculative_tokens=3.
- Cross-check anchor: measured MTP acceptance 62.23%; this is consistent with the previous-generation ShareGPT MTP anchor of 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
