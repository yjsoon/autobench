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
prefill_toks: 31.55
decode_toks: 20.62
mem_gb: 9.52
mem_source: >-
  Shared UMA: baseline host MemAvailable minus the minimum sampled during the run, sampled every 2 seconds; one accelerator summed. amdgpu mem_info_vram_used is a cross-check. This is attributable host-memory delta, not a static engine reservation; startup reservation/profiling is recorded in Notes.
spec_acceptance: >-
  Measured aggregate acceptance rate 70.56% (8148 accepted / 11547 generated); weighted mean acceptance length 2.41; num_speculative_tokens=2.
measured_on: 2026-09-01
completed_at: 2026-09-01 22:15 +0800
engine_image: ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41
source_repo: unsloth/Qwen3.8-27B-GGUF
download_url: https://huggingface.co/unsloth/Qwen3.8-27B-GGUF
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c8 llama q4_k_m mtp2 65536 8 1000 300 256 q4kv
---

## Notes

- Headline: 20.62 aggregate decode tok/s, 31.55 aggregate prefill tok/s, 32 completed, 0 errors, 339.6 s, hit_time_cap=True.
- SLO is median TTFT <= 2000 ms and median TPOT <= 50 ms. TTFT median/p95/p99 = 4892.4/13206.4/15443.8 ms; TPOT median/p95 = 338.6/502.4 ms; goodput = 0.0%.
- Peak shared-memory delta 9.52 GiB from baseline 24792212 kB to minimum 14809204 kB; VRAM cross-check peak None GiB and delta None GiB.
- Checkpoint revision: unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Container behavior flags: --device /dev/kfd, --device /dev/dri, --ipc=host, --shm-size 32g, --security-opt seccomp=unconfined; no extra engine environment variable was set.
- Context-pool adaptation: llama.cpp used -c 65536 as the total context pool with --parallel c; the c8 server measured n_ctx_slot=8192, so context is not 65536 tokens per request at c8. Text-only serving explicitly passed --no-mmproj.
- External recipe source: https://github.com/sudoingX/qwen38-mtp@431bf8a821d49e6ac919febc7a91fb2434318110; file SHA256s: README.md=47dc7f471af4668159359c6f6b14a04c9dc18fd6aa7f1ab223b0efdca110b44; serve_mtp.sh=b2e19c0b0f0b10b8aea9c8601dcc9adeb5dbbe443015895fb5ba698bd7153223; probe.py=d9ea0c9fb10435937c774d4b182a6b5ac8481837e8b3d1aa23738eeb178dfd5f.
- Recipe adaptation: applied --cache-type-k q4_0 --cache-type-v q4_0 -fa 1 --spec-type draft-mtp --spec-draft-n-max 2; the repository's probe.py was not used as the headline harness, because autobench's closed-loop ShareGPT generator supplies the recorded aggregate metrics.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c8, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 16.609 GiB; KV 1.125 GiB; recurrent state 1.184 GiB; total 18.918 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x (18 bytes / 32 values = 0.5625 bytes/value) for q4_0 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - 0.05.476.607 I srv    load_model: initializing, n_slots = 8, n_ctx_slot = 8192, kv_unified = 'false'
  - 0.05.574.974 I srv  llama_server: model loaded
  - 0.05.574.979 I srv  llama_server: listening on http://0.0.0.0:8081
  - 0.11.952.363 I slot print_timing: id  7 | task 0 | draft acceptance = 0.84783 (   39 accepted /    46 generated), mean len =  2.70
  - 0.44.467.647 I slot print_timing: id  7 | task 29 | draft acceptance = 0.64516 (   40 accepted /    62 generated), mean len =  2.29
  - 0.59.446.435 I slot print_timing: id  6 | task 28 | draft acceptance = 0.67000 (   67 accepted /   100 generated), mean len =  2.34
  - 1.07.250.093 I slot print_timing: id  0 | task 34 | draft acceptance = 0.77966 (   92 accepted /   118 generated), mean len =  2.56
  - 1.30.924.897 I slot print_timing: id  3 | task 31 | draft acceptance = 0.91667 (  165 accepted /   180 generated), mean len =  2.83
  - 1.39.518.779 I slot print_timing: id  2 | task 32 | draft acceptance = 0.77889 (  155 accepted /   199 generated), mean len =  2.55
  - 1.45.938.819 I slot print_timing: id  1 | task 33 | draft acceptance = 0.68692 (  147 accepted /   214 generated), mean len =  2.37
  - 1.45.939.451 I slot print_timing: id  5 | task 27 | draft acceptance = 0.69811 (  148 accepted /   212 generated), mean len =  2.40
  - 1.53.983.549 I slot print_timing: id  4 | task 30 | draft acceptance = 0.62115 (  141 accepted /   227 generated), mean len =  2.24
  - 2.06.810.557 I slot print_timing: id  7 | task 69 | draft acceptance = 0.66667 (  128 accepted /   192 generated), mean len =  2.33
  - 2.26.094.255 I slot print_timing: id  6 | task 89 | draft acceptance = 0.75743 (  153 accepted /   202 generated), mean len =  2.51
  - 2.41.820.339 I slot print_timing: id  0 | task 99 | draft acceptance = 0.63393 (  142 accepted /   224 generated), mean len =  2.27
  - 2.52.537.233 I slot print_timing: id  3 | task 131 | draft acceptance = 0.84211 (  160 accepted /   190 generated), mean len =  2.68
  - 3.04.686.846 I slot print_timing: id  1 | task 150 | draft acceptance = 0.87568 (  162 accepted /   185 generated), mean len =  2.74
  - 3.05.335.736 I slot print_timing: id  4 | task 159 | draft acceptance = 0.97110 (  168 accepted /   173 generated), mean len =  2.93
  - 3.12.695.383 I slot print_timing: id  2 | task 142 | draft acceptance = 0.69484 (  148 accepted /   213 generated), mean len =  2.38
  - 3.17.644.009 I slot print_timing: id  5 | task 151 | draft acceptance = 0.72115 (  150 accepted /   208 generated), mean len =  2.44
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median None/None%, temperature peak None C, power median/peak None/None W, sustained clock median None MHz.
- Speculation acceptance: Measured aggregate acceptance rate 70.56% (8148 accepted / 11547 generated); weighted mean acceptance length 2.41; num_speculative_tokens=2.
- Cross-check anchor: measured MTP acceptance 70.56%; this is consistent with the previous-generation ShareGPT MTP anchor of 67% average, per-position 0.84/0.66/0.51, mean length about 3.0 at three draft tokens.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
