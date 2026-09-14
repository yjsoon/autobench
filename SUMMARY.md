# Qwen3.8-27B autobench results

## 1. What was benchmarked

Qwen/Qwen3.8-27B at context 65536, text-only, thinking disabled, ShareGPT fixed max_tokens=256, closed-loop concurrency, actual max_seconds=300 in the recorded commands. The 300-second cap is shorter than autobench's 900-second reference cap and limits external comparison; it is recorded here rather than hidden. Local model revisions and file sizes are on the pages and in the candidate manifest. Dataset SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4 over 94145 conversations; it differs from the task-supplied d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb. The machine-class tag is strix-halo-vulkan. The September rerun adds the qwen38-mtp recipe at commit 431bf8a821d49e6ac919febc7a91fb2434318110, with q4_0 K/V cache and MTP n-max 2/3/4 arms; its repository probe was not used for the headline metrics.

## 2. The machine

See [inventory/machine.json](inventory/machine.json). Memory regime = shared; crash regime = process_oom; native narrow format = BF16 proven; FP8 and native 4-bit execution unproven on this device.

| property | value |
|---|---|
| schema | autobench-adaptation-inventory-2026-08-28 |
| measured_at | 2026-08-28 Asia/Singapore |
| accelerator.vendor | AMD |
| accelerator.count | 1 |
| accelerator.model | AMD Radeon 8060S Graphics |
| accelerator.architecture | gfx1151 / RADV STRIX_HALO |
| accelerator.pci | 1002:1586 |
| accelerator.type | integrated GPU with UMA |
| accelerator.sysfs_memory.vram_total_bytes | 103079215104 |
| accelerator.sysfs_memory.vram_total_gib | 96 |
| accelerator.sysfs_memory.vram_used_idle_bytes | 2842021888 |
| accelerator.sysfs_memory.gtt_total_bytes | 16629108736 |
| accelerator.sysfs_memory.gtt_used_bytes | 148938752 |
| accelerator.vulkan.instance_version | 1.4.357 |
| accelerator.vulkan.api_version | 1.4.354 |
| accelerator.vulkan.driver | RADV |
| accelerator.vulkan.mesa | 26.1.7-arch1.1 |
| accelerator.vulkan.device_local_budget_gib | 72.47 |
| accelerator.vulkan.system_heap_budget_gib | 36.23 |
| accelerator.vulkan.shader_float16 | True |
| accelerator.vulkan.shader_int8 | True |
| accelerator.vulkan.explicit_fp8_extension_observed | False |
| accelerator.clocks_idle.pp_dpm_sclk | 0: 600Mhz; 1: 608Mhz current; 2: 2900Mhz |
| accelerator.clocks_idle.current_clock_hz | 607000000 |
| accelerator.telemetry_idle.gpu_busy_percent | 0 |
| accelerator.telemetry_idle.temperature_mC | 41000 |
| accelerator.telemetry_idle.power_uW | 39768000 |
| accelerator.native_format_probe.bf16_matmul | passed in torch 2.13.0a0+rocm7.13.0a20260422 |
| accelerator.native_format_probe.float8_e4m3_initialization | failed: NotImplementedError: normal_kernel_cuda not implemented for Float8_e4m3fn |
| accelerator.native_format_probe.conclusion | BF16 is the narrowest format proven to execute natively; FP8 and vendor-native 4-bit execution are not proven |
| accelerator.interconnect | n/a; one device |
| host.hostname_redacted | True |
| host.architecture | x86_64 |
| host.cpu | AMD RYZEN AI MAX+ 395 w/ Radeon 8060S |
| host.logical_cpus | 32 |
| host.physical_cores | 16 |
| host.threads_per_core | 2 |
| host.sockets | 1 |
| host.numa_nodes | 1 |
| host.max_mhz | 5187.5 |
| host.kernel | 7.1.8-arch1-3 |
| host.os | Omarchy Linux |
| host.mem_total_kb | 32478728 |
| host.mem_available_idle_kb_snapshot | 20982668 |
| host.swap_total_kb | 64957648 |
| host.swap_free_kb_snapshot | 54067340 |
| host.container_runtime.name | Docker |
| host.container_runtime.client_version | 29.7.2 |
| host.container_runtime.server_version | 29.7.2 |
| host.container_runtime.cgroup_version | 2 |
| host.container_runtime.daemon_architecture | x86_64 |
| host.accelerator_devices | ["/dev/kfd","/dev/dri/renderD128"] |
| host.groups | ["docker","video","render"] |
| storage.model_cache_filesystem | /home |
| storage.mount | /dev/nvme0n1p5 |
| storage.filesystem | btrfs |
| storage.total_bytes | 1045853896704 |
| storage.used_bytes_at_inventory | 485904478208 |
| storage.available_bytes_at_inventory | 557687541760 |
| storage.sequential_direct_read_test | 672837942 bytes in 0.415326 s, 1.6 GB/s |
| software_inside_images.sglang.image | strix-halo-sglang:dev |
| software_inside_images.sglang.image_id | sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272 |
| software_inside_images.sglang.digest_pinned | strix-halo-sglang:dev@sha256:2711177e6d563d227e5eddf894a6069c70c31a7d52c4ddcac97912c935233272 |
| software_inside_images.sglang.pullable | False |
| software_inside_images.sglang.architecture | linux/amd64 |
| software_inside_images.sglang.python | 3.12.13 |
| software_inside_images.sglang.torch | 2.13.0a0+rocm7.13.0a20260422 |
| software_inside_images.sglang.transformers | 5.8.1 |
| software_inside_images.sglang.sglang | 0.0.0.dev1+gb0b8436f1.d20260713 |
| software_inside_images.sglang.hip | 7.13.26154 |
| software_inside_images.vllm.image | kyuz0/vllm-therock-gfx1151:stable |
| software_inside_images.vllm.image_id | sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2 |
| software_inside_images.vllm.digest_pinned | kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2 |
| software_inside_images.vllm.pullable | False |
| software_inside_images.vllm.architecture | linux/amd64 |
| software_inside_images.vllm.python | 3.12.13 |
| software_inside_images.vllm.torch | 2.13.0a0+rocm7.13.0a20260422 |
| software_inside_images.vllm.transformers | 5.5.4 |
| software_inside_images.vllm.vllm | 0.19.2rc1.dev113+g6aa057c9d.d20260422 |
| software_inside_images.vllm.hip | 7.13.26154 |
| software_inside_images.llama.cpp.image | ghcr.io/ggml-org/llama.cpp:full-vulkan |
| software_inside_images.llama.cpp.image_id | sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41 |
| software_inside_images.llama.cpp.digest_pinned | ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41 |
| software_inside_images.llama.cpp.pullable | True |
| software_inside_images.llama.cpp.architecture | linux/amd64 |
| software_inside_images.llama.cpp.version | 9859 (4fc4ec554) |
| software_inside_images.llama.cpp.compiler | GNU 15.2.0 |
| target_model.model | Qwen/Qwen3.8-27B |
| target_model.revision | 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0 |
| target_model.architecture | Qwen3_5ForConditionalGeneration |
| target_model.model_type | qwen3_5 |
| target_model.parameter_count | 27781427952 |
| target_model.parameter_shape | 27B dense |
| target_model.layers | 64 |
| target_model.full_attention_interval | 4 |
| target_model.full_attention_layers | 16 |
| target_model.gated_deltanet_layers | 48 |
| target_model.attention_heads | 24 |
| target_model.kv_heads | 4 |
| target_model.head_dim | 256 |
| target_model.linear_key_heads | 16 |
| target_model.linear_value_heads | 48 |
| target_model.linear_key_head_dim | 128 |
| target_model.linear_value_head_dim | 128 |
| target_model.linear_conv_kernel | 4 |
| target_model.mamba_ssm_dtype | float32 |
| target_model.native_context | 262144 |
| target_model.mtp_num_hidden_layers | 1 |
| target_model.vocab_size | 248320 |
| target_model.tie_word_embeddings | False |
| target_model.modalities | ["text","image","video"] |
| target_model.mm_served | False |
| dataset.path | benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json |
| dataset.size_bytes | 672837942 |
| dataset.sha256 | 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4 |
| dataset.expected_sha256_from_task | d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb |
| dataset.sha256_matches_task_value | False |
| dataset.conversations | 94145 |
| dataset.conversations_with_first_human | 92812 |
| dataset.sampler | first human turn of first N conversations in file order; no shuffle or length filter |
| slo.median_ttft_ms_max | 2000 |
| slo.median_tpot_ms_max | 50 |
| slo.goodput_definition | requests with TTFT <= 2000 ms and TPOT <= 50 ms divided by completed |
| idle_observations.mem_available_kb_samples | [22236004,21852820,22327388,22317012,22145036,20982668] |
| idle_observations.vram_used_bytes_samples | [2842021888,2842079232,2842021888,2842021888,2842021888,2842021888] |
| idle_observations.note | Host MemAvailable varies at idle; each run records its own baseline and 2-second minimum. |
| over_allocation_probe.model | Qwen3.6-35B-A3B-Q4_K_M.gguf |
| over_allocation_probe.attempted_context | [262144,2097152] |
| over_allocation_probe.attempted_parallel | [512,256] |
| over_allocation_probe.observed_process_error | E llama_init_from_model: failed to initialize the context: n_seq_max must be <= 256 |
| over_allocation_probe.observed_followup | n_slots=256 loaded and listened at n_ctx_slot=1024 and n_ctx_slot=8192; no host hard crash |
| over_allocation_probe.crash_regime_evidence | process-contained failure observed; shared UMA still requires watchdog protection |
| autobench_assumptions.memory_regime | shared |
| autobench_assumptions.mem_source_method | Host MemAvailable delta is the headline shared-memory measure, with amdgpu mem_info_vram_used sampled every 2 seconds as a cross-check; one device summed. Engine static reservations are not a footprint and are recorded from startup logs. |
| autobench_assumptions.crash_regime | process_oom |
| autobench_assumptions.watchdog_required | True |
| autobench_assumptions.watchdog_floor_gb | 8 |
| autobench_assumptions.image_arch | linux/amd64 / x86_64 |
| autobench_assumptions.autobench_digests_usable | False |
| autobench_assumptions.native_narrow_format | BF16 proven; FP8 and native 4-bit execution unproven on this device |
| autobench_assumptions.default_quant_format | GGUF Q4_K_M for llama.cpp because it is the only selected format with demonstrated memory headroom; BF16 and official FP8 are separate probes, not assumed native |
| autobench_assumptions.tensor_parallel_applies | False |
| autobench_assumptions.device_count | 1 |
| autobench_assumptions.interconnect_class | n/a |
| autobench_assumptions.memory_ceiling_gb | 96 |
| autobench_assumptions.idle_baseline_gb | 20.01 |
| autobench_assumptions.idle_baseline_definition | latest idle host MemAvailable snapshot in GiB; not the 96 GiB accelerator ceiling |
| autobench_assumptions.gpu_memory_utilization_chosen | 0.8 |
| benchmark_recipe_sources.qwen38_mtp.url | https://github.com/sudoingX/qwen38-mtp |
| benchmark_recipe_sources.qwen38_mtp.revision | 431bf8a821d49e6ac919febc7a91fb2434318110 |
| benchmark_recipe_sources.qwen38_mtp.files_sha256.README.md | 47dc7f471af4668159359c6f6b14a04c9dc18fd6aa7f1ab223b0efdca110b44 |
| benchmark_recipe_sources.qwen38_mtp.files_sha256.serve_mtp.sh | b2e19c0b0f0b10b8aea9c8601dcc9adeb5dbbe443015895fb5ba698bd7153223 |
| benchmark_recipe_sources.qwen38_mtp.files_sha256.probe.py | d9ea0c9fb10435937c774d4b182a6b5ac8481837e8b3d1aa23738eeb178dfd5f |
| benchmark_recipe_sources.qwen38_mtp.adapted_flags | ["--cache-type-k q4_0","--cache-type-v q4_0","-fa 1","--spec-type draft-mtp","--spec-draft-n-max 2\|3\|4"] |
| benchmark_recipe_sources.qwen38_mtp.headline_harness | autobench scripts/bench-serving.py; repository probe.py not used for headline metrics |
| rerun.date | 2026-09-01 |
| rerun.context | 65536 |
| rerun.max_seconds | 300 |
| rerun.max_tokens | 256 |
| rerun.thinking | disabled |
| rerun.configs | ["qwen3-8-27b-q4-k-m-llama-q4kv-c1","qwen3-8-27b-q4-k-m-llama-q4kv-c2","qwen3-8-27b-q4-k-m-llama-q4kv-c4","qwen3-8-27b-q4-k-m-llama-q4kv-c8","qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1","qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c8","qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1","qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c1","qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c2","qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c4","qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c8"] |
| pre_existing_services_observed.container | qwen38 |
| pre_existing_services_observed.image | ghcr.io/ggml-org/llama.cpp:full-vulkan@sha256:fb161d9dd132ecc0878ad8e0170114c269355740c66527f227de0c5f862b5e41 |
| pre_existing_services_observed.started_at | 2026-08-29T10:26:15.288810343Z |
| pre_existing_services_observed.port | 8082 |
| pre_existing_services_observed.model_command | --server --hf-repo unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M --alias qwen3.8-27b -ngl 99 -c 65536 --parallel 1 -cb --reasoning off --host 0.0.0.0 --port 8080 --metrics --log-timestamps --no-mmproj |
| pre_existing_services_observed.observed_at | 2026-09-01 |
| pre_existing_services_observed.observations | ["docker stats reported 0.00% CPU and 44.28 MiB container memory","health returned {\"status\":\"ok\"}","metrics reported requests_processing=0 and requests_deferred=0","no container log lines appeared in the preceding 10 minutes","cumulative metrics at inspection were 17 prompt tokens and 2 generated tokens"] |
| pre_existing_services_observed.disposition | not stopped; treated as an idle background service. Host idle baselines and per-run shared-memory deltas include this background state; no active competing request was observed. |
| autobench_assumptions (whole block) | {"memory_regime":"shared","mem_source_method":"Host MemAvailable delta is the headline shared-memory measure, with amdgpu mem_info_vram_used sampled every 2 seconds as a cross-check; one device summed. Engine static reservations are not a footprint and are recorded from startup logs.","crash_regime":"process_oom","watchdog_required":true,"watchdog_floor_gb":8,"image_arch":"linux/amd64 / x86_64","autobench_digests_usable":false,"native_narrow_format":"BF16 proven; FP8 and native 4-bit execution unproven on this device","default_quant_format":"GGUF Q4_K_M for llama.cpp because it is the only selected format with demonstrated memory headroom; BF16 and official FP8 are separate probes, not assumed native","tensor_parallel_applies":false,"device_count":1,"interconnect_class":"n/a","memory_ceiling_gb":96,"idle_baseline_gb":20.01,"idle_baseline_definition":"latest idle host MemAvailable snapshot in GiB; not the 96 GiB accelerator ceiling","gpu_memory_utilization_chosen":0.8} |

## 3. What had to be adapted from autobench

Reference values were not reused; the substitution column records this machine's evidence.

| assumption | state | substitution |
|---|---|---|
| Total memory ceiling | re-derived | 96 GiB device-local/UMA inventory; not host MemTotal |
| Idle baseline consumed | re-derived | 20.01 GiB latest host MemAvailable snapshot; each run records a new baseline |
| SGLang batch cap | re-derived | runner set --max-running-requests equal to the load level; the SGLang attempt was blocked before a publishable result |
| Usable slack / watchdog | re-derived | 8 GiB floor; watchdog required for shared UMA |
| mem_gb method | re-derived | Host MemAvailable delta is the headline shared-memory measure, with amdgpu mem_info_vram_used sampled every 2 seconds as a cross-check; one device summed. Engine static reservations are not a footprint and are recorded from startup logs. |
| Image architecture/digests | re-derived | linux/amd64 / x86_64; local image digests used where no pullable registry digest exists |
| Native narrow format | re-derived | BF16 proven; FP8 and native 4-bit execution unproven on this device |
| Multiple accelerators / TP | not applicable | one device; no interconnect |
| gpu-memory-utilization | re-derived | 0.8 for ROCm stacks; llama.cpp uses Vulkan allocation |
| Crash regime | re-derived | process_oom with shared UMA watchdog protection |
| Concurrency danger | re-derived | must emerge from this sweep; reference conc-64 not reused |

## 4. Configurations considered

| config | engine | quant | weights_gib | kv_gib_at_ctx | state_gib_at_conc | total_gib | fits | decision | reason |
|---|---|---|---|---|---|---|---|---|---|
| qwen3-8-27b-q4-k-m-llama-c1 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 0.148 | 19.482 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-c2 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 0.296 | 19.630 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-c4 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 0.592 | 19.926 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-c8 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 1.184 | 20.517 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-c16 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 2.367 | 21.701 | yes | blocked | Sweep stopped after the standardized c8 point exceeded the 2x TTFT SLO stop threshold: measured median TTFT 9258.4 ms versus 4000 ms. c16 was not launched; no server error exists for this unattempted rung. |
| qwen3-8-27b-q4-k-m-llama-c32 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 4.734 | 24.068 | yes | blocked | Sweep stopped after the standardized c8 point exceeded the 2x TTFT SLO stop threshold: measured median TTFT 9258.4 ms versus 4000 ms. c32 was not launched; no server error exists for this unattempted required rung. |
| qwen3-8-27b-q4-k-m-llama-c64 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 9.469 | 28.802 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q4-k-m-llama-c128 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 18.938 | 38.271 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q4-k-m-llama-c256 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 37.875 | 57.209 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q4-k-m-llama-n-gram-c1 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 0.148 | 19.482 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-n-gram-c8 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 1.184 | 20.517 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-n-gram-c32 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 4.734 | 24.068 | yes | blocked | Base Q4_K_M llama.cpp c32 was not launched because standardized c8 already exceeded the 2x median-TTFT SLO stop threshold (9258.4 ms > 4000 ms); n-gram c32 has no server error because it was not launched. |
| qwen3-8-27b-q4-k-m-llama-mtp-c1 | llama.cpp | Q4_K_M | 16.609 | 4.000 | 0.148 | 20.757 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-mtp-c8 | llama.cpp | Q4_K_M | 16.609 | 4.000 | 1.184 | 21.793 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-mtp-c32 | llama.cpp | Q4_K_M | 16.609 | 4.000 | 4.734 | 25.344 | yes | blocked | Base Q4_K_M llama.cpp c32 was not launched because standardized c8 already exceeded the 2x median-TTFT SLO stop threshold (9258.4 ms > 4000 ms); MTP c32 has no acceptance or server error because it was not launched. |
| qwen3-8-27b-q8-0-llama-c1 | llama.cpp | Q8_0 | 27.052 | 4.000 | 0.148 | 31.200 | yes | run | measured and published |
| qwen3-8-27b-q8-0-llama-c8 | llama.cpp | Q8_0 | 27.052 | 4.000 | 1.184 | 32.236 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q8-0-llama-c32 | llama.cpp | Q8_0 | 27.052 | 4.000 | 4.734 | 35.787 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q8-0-llama-n-gram-c1 | llama.cpp | Q8_0 | 27.052 | 4.000 | 0.148 | 31.200 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q8-0-llama-n-gram-c8 | llama.cpp | Q8_0 | 27.052 | 4.000 | 1.184 | 32.236 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q8-0-llama-mtp-c1 | llama.cpp | Q8_0 | 28.328 | 4.000 | 0.148 | 32.476 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q8-0-llama-mtp-c8 | llama.cpp | Q8_0 | 28.328 | 4.000 | 1.184 | 33.511 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-bf16-llama-c1 | llama.cpp | BF16 | 50.904 | 4.000 | 0.148 | 55.052 | yes | run | measured and published |
| qwen3-8-27b-bf16-sglang-c1 | SGLang | BF16 | 51.747 | 4.000 | 0.148 | 55.895 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-fp8-sglang-c1 | SGLang | FP8 | 28.747 | 4.000 | 0.148 | 32.895 | yes | blocked | SGLang reported Detected fp8 checkpoint and began loading, but the container exited during the health wait; exact runner result: server did not become healthy for qwen3-8-27b-fp8-sglang-c1 repeat 1. The last saved server line was Using model weights format ['*.safetensors']; no streamed benchmark result was produced. |
| qwen3-8-27b-nvfp4-sglang-c1 | SGLang | NVFP4 | 21.809 | 4.000 | 0.148 | 25.957 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-bf16-vllm-c1 | vLLM | BF16 | 51.747 | 4.000 | 0.148 | 55.895 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-fp8-vllm-c1 | vLLM | FP8 | 28.747 | 4.000 | 0.148 | 32.895 | yes | blocked | First image invocation failed with Docker error exec: Qwen/Qwen3.8-27B-FP8: stat Qwen/Qwen3.8-27B-FP8: no such file or directory. Corrected invocation reached vLLM, then failed: ValueError: Free memory on device cuda:0 (7.34/15.49 GiB) on startup is less than desired GPU memory utilization (0.8, 12.39 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes. |
| qwen3-8-27b-nvfp4-vllm-c1 | vLLM | NVFP4 | 21.809 | 4.000 | 0.148 | 25.957 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-quark-int4-vllm-c1 | vLLM | AMD Quark W4A16 | 18.173 | 4.000 | 0.148 | 22.321 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-quark-mxfp4-vllm-c1 | vLLM | AMD Quark MXFP4 | 18.439 | 4.000 | 0.148 | 22.586 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-bf16-vllm-mtp-c1 | vLLM | BF16 | 51.747 | 4.000 | 0.148 | 55.895 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-fp8-vllm-mtp-c1 | vLLM | FP8 | 28.747 | 4.000 | 0.148 | 32.895 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-nvfp4-vllm-mtp-c1 | vLLM | NVFP4 | 21.809 | 4.000 | 0.148 | 25.957 | yes | skip | planned candidate was not reached; no measurement exists |
| qwen3-8-27b-q4-k-m-llama-eagle-3-c1 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 0.148 | 19.482 | yes | blocked | No Qwen3.8 EAGLE-3 head was found in the current checkpoint/repository inventory; no other model's head was substituted. |
| qwen3-8-27b-q4-k-m-llama-external-draft-c1 | llama.cpp | Q4_K_M | 15.334 | 4.000 | 0.148 | 19.482 | yes | blocked | No architecture-compatible external draft checkpoint was selected; the hybrid GDN/full-attention cache layout makes this path unverified, so it was not run. |
| qwen3-8-27b-fp8-llama-c1 | llama.cpp | FP8 | 28.747 | 4.000 | 0.148 | 32.895 | yes | skip | The selected llama.cpp Vulkan build consumes GGUF; the official FP8 checkpoint is safetensors and no verified FP8 GGUF artifact was selected. |
| qwen3-8-27b-awq-sglang-c1 | SGLang | AWQ | n/a | n/a | n/a | n/a | n/a | skip | No current SGLang support claim for this Qwen3.8 compressed-tensors checkpoint was verified in the selected image; AMD Quark candidates were reserved for vLLM. |
| qwen3-8-27b-awq-exllamav3-c1 | ExLlamaV3 | AWQ | n/a | n/a | n/a | n/a | n/a | blocked | The selected machine path has no verified ROCm/Vulkan ExLlamaV3 backend for this checkpoint; no run was attempted. |
| qwen3-8-27b-q4-k-m-llama-q4kv-c1 | llama.cpp | Q4_K_M | 15.334 | 1.125 | 0.148 | 16.607 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-c8 | llama.cpp | Q4_K_M | 15.334 | 1.125 | 1.184 | 17.642 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 0.148 | 17.882 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c8 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 1.184 | 18.918 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 0.148 | 17.882 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c1 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 0.148 | 17.882 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-c2 | llama.cpp | Q4_K_M | 15.334 | 1.125 | 0.296 | 16.755 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-c4 | llama.cpp | Q4_K_M | 15.334 | 1.125 | 0.592 | 17.051 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c2 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 0.296 | 18.030 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c4 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 0.592 | 18.326 | yes | run | measured and published |
| qwen3-8-27b-q4-k-m-llama-q4kv-c16 | llama.cpp | Q4_K_M | 15.334 | 1.125 | 2.367 | 18.826 | yes | blocked | Base q4_0-KV q8 median TTFT 8728.15 ms exceeded the 2x SLO threshold 4000.00 ms; c16 was not launched. |
| qwen3-8-27b-q4-k-m-llama-q4kv-c32 | llama.cpp | Q4_K_M | 15.334 | 1.125 | 4.734 | 21.193 | yes | blocked | Base q4_0-KV q8 median TTFT 8728.15 ms exceeded the 2x SLO threshold 4000.00 ms; c32 was not launched. |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c16 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 2.367 | 20.101 | yes | blocked | MTP2 q4_0-KV q8 median TTFT 4892.35 ms exceeded the 2x SLO threshold 4000.00 ms; c16 was not launched. |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c32 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 4.734 | 22.469 | yes | blocked | MTP2 q4_0-KV q8 median TTFT 4892.35 ms exceeded the 2x SLO threshold 4000.00 ms; c32 was not launched. |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c16 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 2.367 | 20.101 | yes | blocked | MTP4 q4_0-KV q8 median TTFT 5318.65 ms exceeded the 2x SLO threshold 4000.00 ms; c16 was not launched. |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c32 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 4.734 | 22.469 | yes | blocked | MTP4 q4_0-KV q8 median TTFT 5318.65 ms exceeded the 2x SLO threshold 4000.00 ms; c32 was not launched. |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c8 | llama.cpp | Q4_K_M | 16.609 | 1.125 | 1.184 | 18.918 | yes | run | measured and published |

Table B sizes are computed from current repository file metadata, the 16-layer KV formula and recurrent-state formula. They are screening arithmetic, not measured allocations.

## 5. Results

| slug | engine | speculative | quant | context | concurrency | prefill_toks | decode_toks | ttft_median_ms | ttft_p95_ms | tpot_median_ms | goodput_pct | mem_gb | completed | errors | hit_time_cap | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8-27b-bf16-llama-c1 | llama.cpp | none | BF16 | 65536 | 1 | 1.79 | 4.11 | 2375.2 | 3924.9 | 231.1 | 0.0 | 3.75 | 7 | 0 | True | done |
| qwen3-8-27b-fp8-sglang-c1 | SGLang | none | FP8 | 65536 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-fp8-vllm-c1 | vLLM | none | FP8 | 65536 | 1 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-c1 | llama.cpp | none | Q4_K_M | 65536 | 1 | 2.99 | 11.56 | 1849.0 | 2303.4 | 78.7 | 0.0 | 5.57 | 16 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-c16 | llama.cpp | none | Q4_K_M | 65536 | 16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-c2 | llama.cpp | none | Q4_K_M | 65536 | 2 | 13.49 | 18.72 | 2080.4 | 3011.3 | 92.9 | 0.0 | 8.64 | 26 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-c32 | llama.cpp | none | Q4_K_M | 65536 | 32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-c4 | llama.cpp | none | Q4_K_M | 65536 | 4 | 47.35 | 23.95 | 4803.7 | 9669.3 | 125.2 | 0.0 | 10.13 | 36 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-c8 | llama.cpp | none | Q4_K_M | 65536 | 8 | 57.15 | 29.77 | 9258.4 | 13498.9 | 205.1 | 0.0 | 10.37 | 43 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-mtp-c1 | llama.cpp | MTP | Q4_K_M | 65536 | 1 | 26.30 | 19.19 | 1891.3 | 8477.8 | 40.9 | 67.9 | 9.58 | 28 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-mtp-c32 | llama.cpp | MTP | Q4_K_M | 65536 | 32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-mtp-c8 | llama.cpp | MTP | Q4_K_M | 65536 | 8 | 34.65 | 21.70 | 6073.6 | 14288.5 | 324.6 | 0.0 | 10.42 | 33 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-n-gram-c1 | llama.cpp | n-gram | Q4_K_M | 65536 | 1 | 2.96 | 11.43 | 1847.7 | 2290.9 | 78.8 | 0.0 | 6.01 | 16 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-n-gram-c32 | llama.cpp | n-gram | Q4_K_M | 65536 | 32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-n-gram-c8 | llama.cpp | n-gram | Q4_K_M | 65536 | 8 | 54.28 | 27.57 | 7743.9 | 13201.9 | 229.9 | 0.0 | 11.05 | 42 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-c1 | llama.cpp | none | Q4_K_M | 65536 | 1 | 2.95 | 11.80 | 1713.4 | 2208.8 | 77.7 | 0.0 | 5.54 | 16 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-c16 | llama.cpp | none | Q4_K_M | 65536 | 16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-c2 | llama.cpp | none | Q4_K_M | 65536 | 2 | 16.58 | 18.71 | 1947.2 | 4480.4 | 91.5 | 0.0 | 8.64 | 27 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-c32 | llama.cpp | none | Q4_K_M | 65536 | 32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-c4 | llama.cpp | none | Q4_K_M | 65536 | 4 | 49.38 | 24.82 | 3663.2 | 8981.0 | 122.9 | 0.0 | 9.75 | 36 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-c8 | llama.cpp | none | Q4_K_M | 65536 | 8 | 56.58 | 30.27 | 8728.1 | 12790.4 | 199.3 | 0.0 | 10.35 | 44 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1 | llama.cpp | MTP | Q4_K_M | 65536 | 1 | 17.50 | 19.68 | 1725.8 | 4436.1 | 42.5 | 77.8 | 9.21 | 27 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c16 | llama.cpp | MTP | Q4_K_M | 65536 | 16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c32 | llama.cpp | MTP | Q4_K_M | 65536 | 32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c8 | llama.cpp | MTP | Q4_K_M | 65536 | 8 | 31.55 | 20.62 | 4892.4 | 13206.4 | 338.6 | 0.0 | 9.52 | 32 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1 | llama.cpp | MTP | Q4_K_M | 65536 | 1 | 27.11 | 19.89 | 1725.6 | 8245.2 | 38.6 | 75.0 | 9.28 | 28 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c1 | llama.cpp | MTP | Q4_K_M | 65536 | 1 | 21.97 | 20.16 | 1739.3 | 6349.4 | 40.5 | 74.5 | 9.51 | 28 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c16 | llama.cpp | MTP | Q4_K_M | 65536 | 16 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c2 | llama.cpp | MTP | Q4_K_M | 65536 | 2 | 26.83 | 20.89 | 2120.1 | 8027.9 | 79.2 | 0.0 | 9.63 | 30 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c32 | llama.cpp | MTP | Q4_K_M | 65536 | 32 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c4 | llama.cpp | MTP | Q4_K_M | 65536 | 4 | 35.80 | 21.75 | 2802.8 | 10193.9 | 154.4 | 0.0 | 10.10 | 32 | 0 | True | done |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c8 | llama.cpp | MTP | Q4_K_M | 65536 | 8 | 40.22 | 21.88 | 5318.6 | 16167.7 | 311.9 | 0.0 | 10.38 | 34 | 0 | True | done |
| qwen3-8-27b-q8-0-llama-c1 | llama.cpp | none | Q8_0 | 65536 | 1 | 2.30 | 7.23 | 1883.0 | 2527.3 | 129.8 | 0.0 | 4.21 | 11 | 0 | True | done |

Charts: [throughput_vs_concurrency](charts/throughput_vs_concurrency.png), [ttft_p95_vs_concurrency](charts/ttft_p95_vs_concurrency.png), [pareto_latency_throughput](charts/pareto_latency_throughput.png). Each has a same-basename CSV.

## 6. Speculative decoding

| slug | base_slug | method | num_speculative_tokens | concurrency | acceptance_rate | acceptance_length | decode_toks_spec | decode_toks_base | speedup | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| qwen3-8-27b-q4-k-m-llama-n-gram-c1 | qwen3-8-27b-q4-k-m-llama-c1 | n-gram | n/a (llama.cpp n=12,m=48) | 1 | 12.86 | 5.85 | 11.43 | 11.56 | 0.99 | neutral |
| qwen3-8-27b-q4-k-m-llama-n-gram-c8 | qwen3-8-27b-q4-k-m-llama-c8 | n-gram | n/a (llama.cpp n=12,m=48) | 8 | 23.58 | 10.44 | 27.57 | 29.77 | 0.93 | loss |
| qwen3-8-27b-q4-k-m-llama-n-gram-c32 | qwen3-8-27b-q4-k-m-llama-c32 | n-gram | n/a (llama.cpp n=12,m=48) | 32 | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-mtp-c1 | qwen3-8-27b-q4-k-m-llama-c1 | MTP | 3 | 1 | 60.97 | 2.82 | 19.19 | 11.56 | 1.66 | win |
| qwen3-8-27b-q4-k-m-llama-mtp-c8 | qwen3-8-27b-q4-k-m-llama-c8 | MTP | 3 | 8 | 60.74 | 2.82 | 21.70 | 29.77 | 0.73 | loss |
| qwen3-8-27b-q4-k-m-llama-mtp-c32 | qwen3-8-27b-q4-k-m-llama-c32 | MTP | 3 | 32 | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1 | qwen3-8-27b-q4-k-m-llama-q4kv-c1 | MTP | 2 | 1 | 72.80 | 2.45 | 19.68 | 11.80 | 1.67 | win |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c8 | qwen3-8-27b-q4-k-m-llama-q4kv-c8 | MTP | 2 | 8 | 70.56 | 2.41 | 20.62 | 30.27 | 0.68 | loss |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1 | qwen3-8-27b-q4-k-m-llama-q4kv-c1 | MTP | 3 | 1 | 62.23 | 2.86 | 19.89 | 11.80 | 1.69 | win |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c1 | qwen3-8-27b-q4-k-m-llama-q4kv-c1 | MTP | 4 | 1 | 53.69 | 3.14 | 20.16 | 11.80 | 1.71 | win |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c2 | qwen3-8-27b-q4-k-m-llama-q4kv-c2 | MTP | 4 | 2 | 52.55 | 3.09 | 20.89 | 18.71 | 1.12 | win |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c4 | qwen3-8-27b-q4-k-m-llama-q4kv-c4 | MTP | 4 | 4 | 53.06 | 3.11 | 21.75 | 24.82 | 0.88 | loss |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c16 | qwen3-8-27b-q4-k-m-llama-q4kv-c16 | MTP | 2 | 16 | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c32 | qwen3-8-27b-q4-k-m-llama-q4kv-c32 | MTP | 2 | 32 | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c16 | qwen3-8-27b-q4-k-m-llama-q4kv-c16 | MTP | 4 | 16 | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c32 | qwen3-8-27b-q4-k-m-llama-q4kv-c32 | MTP | 4 | 32 | n/a | n/a | n/a | n/a | n/a | blocked |
| qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c8 | qwen3-8-27b-q4-k-m-llama-q4kv-c8 | MTP | 4 | 8 | 52.23 | 3.08 | 21.88 | 30.27 | 0.72 | loss |

67% anchor cross-check: measured MTP acceptance median 60.74%; this is within 10 points of the 67% previous-generation ShareGPT MTP anchor; n-gram is method-specific and excluded.
Per-method crossover below 1.0: MTP first loss at c4; n-gram first loss at c1.

MTP and n-gram pages are in addition to their same-engine/quant/concurrency base siblings. [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [SGLang speculative decoding](https://sgl-project.github.io/advanced_features/speculative_decoding.html), [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), and [AMD Quark](https://github.com/amd/Quark).

Charts: [speculative_speedup_vs_concurrency](charts/speculative_speedup_vs_concurrency.png) and [acceptance_vs_concurrency](charts/acceptance_vs_concurrency.png).

## 7. Memory

See [memory_headroom](charts/memory_headroom.png). mem_gb is a shared-UMA host MemAvailable delta, not a static engine footprint. Startup reservation/weight/KV evidence is quoted on each page; missing engine breakdowns are marked not emitted.

## 8. Failures and blocks

- qwen3-8-27b-q4-k-m-llama-c16 — Sweep stopped after the standardized c8 point exceeded the 2x TTFT SLO stop threshold: measured median TTFT 9258.4 ms versus 4000 ms. c16 was not launched; no server error exists for this unattempted rung.
- qwen3-8-27b-q4-k-m-llama-c32 — Sweep stopped after the standardized c8 point exceeded the 2x TTFT SLO stop threshold: measured median TTFT 9258.4 ms versus 4000 ms. c32 was not launched; no server error exists for this unattempted required rung.
- qwen3-8-27b-q4-k-m-llama-n-gram-c32 — Base Q4_K_M llama.cpp c32 was not launched because standardized c8 already exceeded the 2x median-TTFT SLO stop threshold (9258.4 ms > 4000 ms); n-gram c32 has no server error because it was not launched.
- qwen3-8-27b-q4-k-m-llama-mtp-c32 — Base Q4_K_M llama.cpp c32 was not launched because standardized c8 already exceeded the 2x median-TTFT SLO stop threshold (9258.4 ms > 4000 ms); MTP c32 has no acceptance or server error because it was not launched.
- qwen3-8-27b-fp8-sglang-c1 — SGLang reported Detected fp8 checkpoint and began loading, but the container exited during the health wait; exact runner result: server did not become healthy for qwen3-8-27b-fp8-sglang-c1 repeat 1. The last saved server line was Using model weights format ['*.safetensors']; no streamed benchmark result was produced.
- qwen3-8-27b-fp8-vllm-c1 — First image invocation failed with Docker error exec: Qwen/Qwen3.8-27B-FP8: stat Qwen/Qwen3.8-27B-FP8: no such file or directory. Corrected invocation reached vLLM, then failed: ValueError: Free memory on device cuda:0 (7.34/15.49 GiB) on startup is less than desired GPU memory utilization (0.8, 12.39 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
- qwen3-8-27b-q4-k-m-llama-q4kv-c16 — Base q4_0-KV q8 median TTFT 8728.15 ms exceeded the 2x SLO threshold 4000.00 ms; c16 was not launched.
- qwen3-8-27b-q4-k-m-llama-q4kv-c32 — Base q4_0-KV q8 median TTFT 8728.15 ms exceeded the 2x SLO threshold 4000.00 ms; c32 was not launched.
- qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c16 — MTP2 q4_0-KV q8 median TTFT 4892.35 ms exceeded the 2x SLO threshold 4000.00 ms; c16 was not launched.
- qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c32 — MTP2 q4_0-KV q8 median TTFT 4892.35 ms exceeded the 2x SLO threshold 4000.00 ms; c32 was not launched.
- qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c16 — MTP4 q4_0-KV q8 median TTFT 5318.65 ms exceeded the 2x SLO threshold 4000.00 ms; c16 was not launched.
- qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c32 — MTP4 q4_0-KV q8 median TTFT 5318.65 ms exceeded the 2x SLO threshold 4000.00 ms; c32 was not launched.

## 9. Recommendation

- Best aggregate throughput: qwen3-8-27b-q4-k-m-llama-q4kv-c8 at c8, 30.27 aggregate decode tok/s.
- Best single-user latency: qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1 at c1, 38.6 ms median TPOT and 1725.6 ms median TTFT.
- Best goodput: qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1 at 77.8%, with the SLO restated as median TTFT <= 2000 ms and median TPOT <= 50 ms.
- Fastest-engine comparison: no fastest engine can be established; qwen3-8-27b-fp8-sglang-c1 and qwen3-8-27b-fp8-vllm-c1 are blocked before valid throughput, so no speed ratio or equivalence claim is made.
- Smallest viable model-file set: qwen3-8-27b-q4-k-m-llama-c1 at 15.334 GiB of measured checkpoint files plus any selected draft sidecar; llama.cpp emitted no numeric engine weight breakdown, so mem_gb is not used as a footprint comparison.
- Speculation is per concurrency: qwen3-8-27b-q4-k-m-llama-n-gram-c1 c1 speedup 0.99x acceptance 12.86%; qwen3-8-27b-q4-k-m-llama-n-gram-c8 c8 speedup 0.93x acceptance 23.58%; qwen3-8-27b-q4-k-m-llama-mtp-c1 c1 speedup 1.66x acceptance 60.97%; qwen3-8-27b-q4-k-m-llama-mtp-c8 c8 speedup 0.73x acceptance 60.74%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c1 c1 speedup 1.67x acceptance 72.80%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp2-c8 c8 speedup 0.68x acceptance 70.56%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp3-c1 c1 speedup 1.69x acceptance 62.23%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c1 c1 speedup 1.71x acceptance 53.69%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c2 c2 speedup 1.12x acceptance 52.55%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c4 c4 speedup 0.88x acceptance 53.06%; qwen3-8-27b-q4-k-m-llama-q4kv-mtp4-c8 c8 speedup 0.72x acceptance 52.23%
- Recommendation is constrained to the measured Pareto frontier: qwen3-8-27b-q4-k-m-llama-q4kv-c8.

For ShareGPT general chat with fixed 256-token responses, run llama.cpp Q4_K_M none at concurrency 8 (slug qwen3-8-27b-q4-k-m-llama-q4kv-c8), giving 30.27 tok/s aggregate and 199.3 ms per token per user at 10.35 GB. Above c8, no higher successful rung was measured; the next required sweep rung is represented by a blocked page.

## 10. What is not known

- Candidate rows marked skip/blocked were not publishable runs; no number is inferred for them.
- The local ShareGPT checksum differs from the task-supplied checksum; only the local SHA256 proves these runs used the same file.
- Local SGLang/vLLM images are not pullable registry artifacts; external reproduction needs their build inputs.
- llama.cpp did not emit a numeric KV/weight/activation/non-torch breakdown for the completed Q4 run; slot count and n_ctx_slot are measured and KV arithmetic is computed.
- Sustained clock fields were blank through the sysfs sampler; busy, temperature and power were sampled.
- A pre-existing qwen38 llama.cpp container was loaded before the rerun and left running. It was observed idle (0.00% CPU, zero processing/deferred requests, no recent log activity); its background state is included in the baselines, and no active competing request was observed.
- No matching Qwen3.8 EAGLE-3 head was found; no other model's head was substituted. External-draft compatibility was not published as success.
- Documentation claims remain claims to recheck: [vLLM speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/), [SGLang speculative decoding](https://sgl-project.github.io/advanced_features/speculative_decoding.html), [llama.cpp server options](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md), and [AMD Quark](https://github.com/amd/Quark).
- ExLlamaV3, archived TGI, NVIDIA-only stacks and unverified vendor stacks were not run on this AMD Vulkan/ROCm machine.

