---
title: Gemma 4 31B · vLLM · NVFP4
model: google/gemma-4-31B-it
company: Google
family: Gemma
params: 33B (dense)
engine: vLLM
quant: NVFP4
quant_rationale: NVIDIA's own NVFP4 build (TensorRT-Model-Optimizer / modelopt) — Blackwell-native 4-bit, the format that gave the Nemotron headliners their edge. Chosen to rescue the 31B from its catastrophic llama.cpp KV-cliff result.
source_repo: nvidia/Gemma-4-31B-IT-NVFP4
download_url: https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4
context: 65536
modalities: [text, image]
mm_served: false
tags: [gemma-4-31b, Google, Gemma, NVFP4, 16-40B]
status: done
prefill_toks: 182.75
decode_toks: 166.96
mem_gb: 109.83
mem_source: system MemAvailable delta (10s sampling) — vLLM static KV reservation (util 0.85), see Notes
measured_on: 2026-06-22
completed_at: 2026-06-22 09:12 +08
run_command: |
  # vllm/vllm-openai:cu130-nightly (ENTRYPOINT ["vllm","serve"]). NVIDIA NVFP4 (modelopt).
  docker run -d --gpus all --ipc=host -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface --env HF_TOKEN=*** \
    vllm/vllm-openai:cu130-nightly nvidia/Gemma-4-31B-IT-NVFP4 \
    --host 0.0.0.0 --port 8000 --max-model-len 65536 \
    --gpu-memory-utilization 0.85 --max-num-seqs 32
  python3 scripts/bench-serving.py --base-url http://localhost:8000 \
    --model nvidia/Gemma-4-31B-IT-NVFP4 \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json \
    --num-prompts 1000 --max-seconds 900 --concurrency 32 --max-tokens 256
---

**NVFP4 on vLLM rescues the Gemma-31B — 2× the throughput of the llama.cpp path, and zero errors.**
Google's dense Gemma-4-31B in NVIDIA's own NVFP4 build, on the cu130-nightly vLLM.

- **Workload:** ShareGPT V3, concurrency 32. **672/1000, 0 errors** — still **hit the 15-min cap**
  (Gemma stays compute-heavy), but more than double the completions of the GGUF run. Loaded +
  CUDA-graph captured in **413 s**.
- **Throughput (aggregate, conc 32):** prefill **182.8 tok/s**, decode **167.0 tok/s**. TTFT median
  **531 ms**, TPOT median **176 ms** (≈5.7 tok/s/stream), req throughput 0.72/s.
- **The headline comparison — same model, two engines/quants:**

  | Config | prefill | decode | completed | errors | TTFT med |
  |---|---|---|---|---|---|
  | llama.cpp Q4_K_M | 67.8 | 78.5 | 322/1000 | 10 | 13.0 s |
  | **vLLM NVFP4** | **182.8** | **167.0** | **672/1000** | **0** | **0.53 s** |

  **~2.1× decode, ~2.7× prefill, 2× the throughput-limited completions, and a 25× better TTFT** (13 s →
  0.5 s). The GGUF run drowned in the global-attention KV cliff (88 GB resident, per-token decode
  crawling); vLLM's paged KV + Blackwell-native NVFP4 weights cut both the weight traffic and the KV
  pressure, so it stays responsive. This is the single biggest engine/quant swing in the benchmark and
  the clearest argument for NVFP4-on-vLLM for the heavy dense Gemmas.
- **Still cap-bound, though.** Even rescued, Gemma-4-31B is compute-heavy enough to miss 1000 prompts in
  15 min at conc 32 — it's a genuinely expensive dense model, just no longer pathological. The FP8 and
  EAGLE3 variants test whether quant precision or speculation closes the rest of the gap.
- **Memory: 109.8 GB is the vLLM `--gpu-memory-utilization 0.85` reservation,** not the footprint
  (NVFP4 weights ≈ 17 GB). The real KV cliff is absorbed into the reservation here — vLLM sizes the KV
  pool to fit, where llama.cpp pre-allocated it raw to 88 GB.
- Text path benchmarked (`mm_served: false`).
