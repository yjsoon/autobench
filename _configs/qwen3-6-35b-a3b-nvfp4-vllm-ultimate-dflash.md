---
title: Qwen3.6-35B-A3B · vLLM-ultimate (AEON) · NVFP4 + DFlash · conc 1/8/32
model: nvidia/Qwen3.6-35B-A3B-NVFP4
company: Alibaba
family: Qwen
params: 35B / 3B (MoE, hybrid GDN+full-attn) + DFlash external drafter
engine: vLLM (aeon-vllm-ultimate custom container, v0.23.0+aeon.sm121a.dflash)
speculative: DFlash (z-lab/Qwen3.6-35B-A3B-DFlash @31977fbe small-page rev, num_speculative_tokens 11)
quant: NVFP4 (modelopt_mixed — W4A16_NVFP4 experts + FP8 GDN gates)
quant_rationale: The PRODUCTION checkpoint (the one llm.manek.sg actually serves) under DFlash spec-decode, to test whether the AEON DFlash win transfers off the abliterated heretic onto the official ModelOpt weights — and whether the FP8-quantized GDN gates nose-dive draft acceptance (Vassallo). Loaded the production way (`--quantization modelopt --moe-backend marlin`, NO `VLLM_TEST_FORCE_FP8_MARLIN` — that env crashes the FP8 dense layers via an `AttributeError` on `orig_dtype`). SAFETY — untrusted third-party image; run with NO credentials, official weights mounted READ-ONLY from the HF cache, drafter READ-ONLY, port loopback-only.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 40960
modalities: [text, image, video]
mm_served: false
concurrency: 1
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-1]
status: done
prefill_toks: 118.94
decode_toks: 101.9
mem_gb: 111.0
mem_source: system `free -h` after load at util 0.85 (111 GiB used / ~10 GiB free); GPU KV cache 1,033,297 tokens
spec_acceptance: ~20–30% avg draft acceptance · mean acceptance length ~3.2–4.3 of 11 drafted · per-position ~0.72–0.81 (pos0) decaying to ~0.03–0.07 (pos10)
measured_on: 2026-06-28
completed_at: 2026-06-28 21:30 +0800
engine_image: ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix@sha256:be9e05a11da6e72607ab6f3e960993b253b673af0727005122a3266129a518e3
run_command: |
  # UNTRUSTED image — NO creds; official weights + drafter READ-ONLY; loopback port. Boots once, sweeps conc 1/8/32.
  # The small-page drafter (@31977fbe: 4 kv-heads, sliding_window null) is what clears the page-size assert;
  # AEON's REQUIRED post-04-19 main drafter (sw 4096 / 8 kv-heads) asserts on this box. Mount the model's
  # cache ROOT (snapshot symlinks point into ../../blobs), serve the snapshot subdir.
  docker run -d --name serving-official --gpus all --ipc=host \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e ENABLE_NVFP4_SM100=0 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4:/officialmodel:ro \
    -v ~/ornith/drafter-smallpage:/drafter:ro -p 127.0.0.1:8000:8000 \
    --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix \
    serve /officialmodel/snapshots/<snap> --served-model-name official --host 0.0.0.0 --port 8000 \
    --quantization modelopt --moe-backend marlin --trust-remote-code --attention-backend flash_attn \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --max-model-len 40960 --gpu-memory-utilization 0.85 --enable-chunked-prefill --enable-prefix-caching \
    --max-num-seqs 64 --max-num-batched-tokens 32768 \
    --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":11}'
  python3 scripts/bench-serving.py --base-url http://127.0.0.1:8000 --model official \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json --concurrency {1,8,32} --max-tokens 256
---

**DONE — DFlash boots and serves clean on the production checkpoint, but loses to native MTP except at conc-1.**
Official `nvidia/Qwen3.6-35B-A3B-NVFP4` (modelopt_mixed: W4A16_NVFP4 experts + FP8 GDN gates) on the AEON image,
ctx 40960, DFlash n=11 via the small-page drafter, util 0.85. **0 errors at every concurrency.**

| conc | prefill tok/s | decode tok/s (agg) | vs MTP decode | TTFT median |
|------|--------------:|-------------------:|--------------:|------------:|
| 1    | 118.94 | **101.9** | **+2.9%** (matched MTP 99.04) † | 2472 ms |
| 8    | 273.85 | 269.9 | −6.6% (MTP 289.1) | 7415 ms |
| 32   | 400.87 | 401.2 | −25.9% (MTP 541.3) | 19972 ms |

† **conc-1 vs-MTP corrected 2026-07-01:** originally quoted **+8.5%** against the published MTP c1 (93.9), but
that was a 300 s-cap run; a **matched 600 s-cap MTP c1 recheck = 99.04 tok/s**, shrinking DFlash's edge to
**~+2.9%** — and even that is ctx-confounded (DFlash ctx 40960 vs MTP 65536). So DFlash's single-stream
advantage is marginal, essentially a wash, reinforcing the "keep MTP" verdict below.

- **DFlash wins single-stream, loses batched.** The only win is conc-1 (~+2.9% vs matched-cap MTP; +8.5% vs the
  original short-cap MTP — see † above) — and even that is partly the shorter
  ctx (40960 vs the MTP baseline's 65536; though ShareGPT sequences are short enough that the gap is mostly real).
  At conc-8/32 it falls behind, sharply at 32. Classic spec-decode shape: it helps the bandwidth-bound single stream and
  hurts the compute-bound batch.
- **Why MTP wins under load — draft efficiency.** MTP drafts **3** tokens at **~66% acceptance / accept-len ~3.0-of-3**
  (almost no wasted compute). DFlash drafts **11** at **~25% / accept-len ~3.7-of-11** → ≈7 wasted drafter forward
  passes per step. On this compute-bound MoE that wasted work directly sinks aggregate throughput as streams divide the GPU.
- **The FP8-gate nose-dive did NOT happen.** Acceptance (~20–30%) is *slightly higher* than the BF16-gate heretic — the
  z-lab drafter was trained against the true Qwen3.6-35B-A3B base that this checkpoint quantizes, so it aligns better
  here than with the abliterated heretic. Vassallo's "~4 of 15" collapse is overstated for this image.
- **Boot / kernel:** `modelopt_mixed`; **MARLIN** NVFP4 MoE + `FlashInferFP8ScaledMMLinearKernel` for the FP8 dense
  layers; FLASH_ATTN; attention block size **1136** (unifies the draft's small page with the mamba/GDN page). GPU KV
  cache **1,033,297 tokens**; ~111 GiB used at util 0.85. Boot ~390 s.
- **Loader gotcha:** AEON's heretic recipe sets `VLLM_TEST_FORCE_FP8_MARLIN=1`; on this ModelOpt checkpoint that forces
  the FP8 GDN-gate linears through a Marlin prep path that crashes (`'MergedColumnParallelLinear' object has no attribute
  'orig_dtype'`). Drop the env and use the production `--quantization modelopt --moe-backend marlin` instead.
- **TPOT 0.0** is the usual `qwen3` reasoning-parser client artifact — decode tok/s is the real number.

**Trusted recommendation: stay on native MTP** (`qwen3-6-35b-a3b-nvfp4-vllm-mtp-c1`/`-c8`/maxctx). DFlash *works* on
the production model now, but the only gain is a modest single-stream edge bought with an external drafter, a
drafter revision AEON explicitly forbids, and an untrusted image replacing the pinned vLLM. Not worth it for a
mixed-workload gateway. Cross-ref: [`…heretic…dflash`](qwen3-6-35b-a3b-heretic-aeon-vllm-ultimate-dflash),
[`…dflash-blocked`](ornith-1-0-35b-aeon-vllm-nvfp4-dflash-blocked), `notes/INCOMPATIBILITIES.md`.
