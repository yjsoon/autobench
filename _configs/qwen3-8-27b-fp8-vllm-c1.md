---
title: Qwen3.8-27B · vLLM · FP8
model: Qwen/Qwen3.8-27B
company: Alibaba
family: Qwen
params: 27B (dense)
engine: vLLM
speculative: none
quant: FP8
quant_rationale: >-
  FP8 is the official Qwen3.8 W8A8 checkpoint. No explicit Vulkan FP8 extension was observed, so it is tested only through stacks that claim a current FP8 path and the startup backend is recorded.
context: 65536
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.8-27b, Alibaba, Qwen, FP8, 16-40B, conc-1, strix-halo-vulkan]
status: blocked
prefill_toks: n/a
decode_toks: n/a
mem_gb: n/a
mem_source: >-
  Shared UMA method would use baseline host MemAvailable minus the 2-second minimum, with one accelerator summed. No publishable peak was recorded for this blocked attempt; static reservations are not a footprint.
spec_acceptance: >-
  Not applicable on a base page.
measured_on: 2026-08-28
completed_at: 2026-08-28 17:13 +0800
engine_image: kyuz0/vllm-therock-gfx1151:stable@sha256:f89c8c689ade28877ade980ba0f29b3142af16c6ebb7f3f285311d38bc81a8a2
source_repo: Qwen/Qwen3.8-27B-FP8
download_url: https://huggingface.co/Qwen/Qwen3.8-27B-FP8
run_command: |
  scripts/run-qwen38-config.sh qwen3-8-27b-fp8-vllm-c1 vllm fp8 base 65536 1 1000 300 256
---

## Notes

- Status: blocked. Exact observed error/reason: First image invocation failed with Docker error exec: Qwen/Qwen3.8-27B-FP8: stat Qwen/Qwen3.8-27B-FP8: no such file or directory. Corrected invocation reached vLLM, then failed: ValueError: Free memory on device cuda:0 (7.34/15.49 GiB) on startup is less than desired GPU memory utilization (0.8, 12.39 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
- Peak shared-memory delta 2.37 GiB from baseline 22024008 kB to minimum 19534088 kB; VRAM cross-check peak 2.65 GiB and delta 0.01 GiB.
- Checkpoint revision: Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a; base Qwen3.8-27B config revision: 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0.
- Behavior-changing container environment: HSA_OVERRIDE_GFX_VERSION=11.5.1.
- Workload uses the first human turn of the first N ShareGPT conversations with one, file order, no shuffle or length filter. Local SHA256 is 35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4; the task-supplied SHA256 d7094af68bb58022696728841937d7285db423eba9d055b3386797b28db2cbdb differs. max_tokens=256; thinking disabled uniformly.
- TTFT is first streamed content token, skipping leading role/empty chunks. TPOT excludes the first completion token. prefill_toks/decode_toks are whole-server aggregates; at c1, per-user decode is aggregate decode divided by the in-flight count.
- Computed fit arithmetic: selected model files (including any draft sidecar where applicable) 28.747 GiB; KV 4.000 GiB; recurrent state 0.148 GiB; total 32.895 GiB; fit=yes. KV formula: 2 x 16 x 4 x 256 x 2 bytes/token for BF16 K/V; GDN recurrent state is 48 x 48 x 128 x 128 x 4 bytes plus approximately 7.5 MiB convolution state per sequence.
- Engine startup evidence (measured log lines):
  - (APIServer pid=1) INFO 08-28 09:12:10 [utils.py:233] non-default args: {'model_tag': 'Qwen/Qwen3.8-27B-FP8', 'host': '0.0.0.0', 'model': 'Qwen/Qwen3.8-27B-FP8', 'trust_remote_code': True, 'revision': '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a', 'max_model_len': 65536, 'enforce_eager': True, 'served_model_name': ['Qwen/Qwen3.8-27B'], 'gpu_memory_utilization': 0.8, 'max_num_seqs': 1}
  - (APIServer pid=1) INFO 08-28 09:12:32 [nixl_utils.py:20] Setting UCX_RCACHE_MAX_UNRELEASED to '1024' to avoid a rare memory leak in UCX when using NIXL.
  - (APIServer pid=1) The `use_fast` parameter is deprecated and will be removed in a future version. Use `backend="torchvision"` instead of `use_fast=True`, or `backend="pil"` instead of `use_fast=False`.
  - (EngineCore pid=509) INFO 08-28 09:12:56 [core.py:107] Initializing a V1 LLM engine (v0.19.2rc1.dev113+g6aa057c9d.d20260422) with config: model='Qwen/Qwen3.8-27B-FP8', speculative_config=None, tokenizer='Qwen/Qwen3.8-27B-FP8', skip_tokenizer_init=False, tokenizer_mode=auto, revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, tokenizer_revision=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, trust_remote_code=True, dtype=torch.bfloat16, max_seq_len=65536, download_dir=None, load_format=auto, tensor_parallel_size=1, pipeline_parallel_size=1, data_parallel_size=1, decode_context_parallel_size=1, dcp_comm_backend=ag_rs, disable_custom_all_reduce=True, quantization=fp8, quantization_config=None, enforce_eager=True, enable_return_routed_experts=False, kv_cache_dtype=auto, device_config=cuda, structured_outputs_config=StructuredOutputsConfig(backend='auto', disable_any_whitespace=False, disable_additional_properties=False, reasoning_parser='', reasoning_parser_plugin='', enable_in_reasoning=False), observability_config=ObservabilityConfig(show_hidden_metrics_for_version=None, otlp_traces_endpoint=None, collect_detailed_traces=None, kv_cache_metrics=False, kv_cache_metrics_sample=0.01, cudagraph_metrics=False, enable_layerwise_nvtx_tracing=False, enable_mfu_metrics=False, enable_mm_processor_stats=False, enable_logging_iteration_details=False), seed=0, served_model_name=Qwen/Qwen3.8-27B, enable_prefix_caching=False, enable_chunked_prefill=True, pooler_config=None, compilation_config={'mode': <CompilationMode.NONE: 0>, 'debug_dump_path': None, 'cache_dir': '', 'compile_cache_save_format': 'binary', 'backend': 'inductor', 'custom_ops': ['+quant_fp8', '+sparse_attn_indexer', 'all', '+quant_fp8'], 'ir_enable_torch_wrap': False, 'splitting_ops': [], 'compile_mm_encoder': False, 'cudagraph_mm_encoder': False, 'encoder_cudagraph_token_budgets': [], 'encoder_cudagraph_max_vision_items_per_batch': 0, 'encoder_cudagraph_max_frames_per_batch': None, 'compile_sizes': [], 'compile_ranges_endpoints': [2048], 'inductor_compile_config': {'enable_auto_functionalized_v2': False, 'combo_kernels': True, 'benchmark_combo_kernel': True}, 'inductor_passes': {}, 'cudagraph_mode': <CUDAGraphMode.NONE: 0>, 'cudagraph_num_of_warmups': 0, 'cudagraph_capture_sizes': [], 'cudagraph_copy_inputs': False, 'cudagraph_specialize_lora': True, 'use_inductor_graph_partition': False, 'pass_config': {'fuse_norm_quant': True, 'fuse_act_quant': True, 'fuse_attn_quant': False, 'enable_sp': False, 'fuse_gemm_comms': False, 'fuse_allreduce_rms': False, 'fuse_act_padding': False, 'fuse_mla_dual_rms_norm': False, 'fuse_rope_kvcache': False}, 'max_cudagraph_capture_size': 0, 'dynamic_shapes_config': {'type': <DynamicShapesType.BACKED: 'backed'>, 'evaluate_guards': False, 'assume_32_bit_indexing': False}, 'local_cache_dir': None, 'fast_moe_cold_start': False, 'static_all_moe_layers': []}, kernel_config=KernelConfig(ir_op_priority=IrOpPriorityConfig(rms_norm=['vllm_c', 'native']), enable_flashinfer_autotune=True, moe_backend='auto')
  - (EngineCore pid=509) INFO 08-28 09:12:58 [parallel_state.py:1402] world_size=1 rank=0 local_rank=0 distributed_init_method=tcp://172.17.0.2:44261 backend=nccl
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132] EngineCore failed to start.
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132] Traceback (most recent call last):
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]   File "/opt/venv/lib64/python3.12/site-packages/vllm/v1/engine/core.py", line 1106, in run_engine_core
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]     engine_core = EngineCoreProc(*args, engine_index=dp_rank, **kwargs)
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]   File "/opt/venv/lib64/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]     return func(*args, **kwargs)
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]            ^^^^^^^^^^^^^^^^^^^^^
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]   File "/opt/venv/lib64/python3.12/site-packages/vllm/v1/engine/core.py", line 872, in __init__
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]     super().__init__(
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]   File "/opt/venv/lib64/python3.12/site-packages/vllm/v1/engine/core.py", line 116, in __init__
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]     self.model_executor = executor_class(vllm_config)
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]   File "/opt/venv/lib64/python3.12/site-packages/vllm/tracing/otel.py", line 178, in sync_wrapper
  - (EngineCore pid=509) ERROR 08-28 09:12:59 [core.py:1132]     return func(*args, **kwargs)
- Engine allocation cross-check: the saved logs emitted no numeric KV-cache size, weight/activation/non-torch breakdown, or engine-reported maximum concurrency; n_slots/n_ctx_slot and max_running_requests are configured settings, while the KV/state totals above are computed.
- Telemetry: busy peak/median 0.0/0.0%, temperature peak 48.0 C, power median/peak 25.128999999999998/30.942 W, sustained clock median None MHz.
- Adaptation: shared UMA and one accelerator were measured here; the watchdog floor is the machine-specific 8 GiB safety floor; local amd64 images are pinned by local digest; BF16 is the only narrow format proven by the native probe; TP is not applicable. Recipe tag strix-halo-vulkan is defined once in SUMMARY.md.
- Machine-state caveat: pre-existing container qwen38 was loaded before this run. It was observed idle (0.00% CPU, requests_processing=0, requests_deferred=0, and no log activity in the preceding 10 minutes); it was not stopped. The idle baseline and shared-memory deltas therefore include that background state, but no active competing request was observed.
- Candidate note: First image invocation failed with Docker error exec: Qwen/Qwen3.8-27B-FP8: stat Qwen/Qwen3.8-27B-FP8: no such file or directory. Corrected invocation reached vLLM, then failed: ValueError: Free memory on device cuda:0 (7.34/15.49 GiB) on startup is less than desired GPU memory utilization (0.8, 12.39 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes.
