---
title: Qwen3.6-35B-A3B · vLLM-ultimate (AEON) · NVFP4 + DFlash · conc 2
model: nvidia/Qwen3.6-35B-A3B-NVFP4
company: Alibaba
family: Qwen
params: 35B / 3B (MoE, hybrid GDN+full-attn) + DFlash external drafter
engine: vLLM (aeon-vllm-ultimate custom container, v0.23.0+aeon.sm121a.dflash)
speculative: DFlash (z-lab/Qwen3.6-35B-A3B-DFlash @31977fbe small-page rev, num_speculative_tokens 11)
quant: NVFP4 (modelopt_mixed — W4A16_NVFP4 experts + FP8 GDN gates)
quant_rationale: conc-2 fine-grained point of the DFlash money-chart line (EXPERIMENTS.md #13/#16). The published DFlash series measured conc-1/8/32 only; this pins the 1→2 region against the matched MTP sweep to locate the exact crossover. Same one-boot protocol as the published series (official checkpoint, small-page drafter, max-num-seqs 64, ctx 40960). SAFETY — untrusted third-party image; run with NO credentials, official weights + drafter READ-ONLY, port loopback-only.
source_repo: nvidia/Qwen3.6-35B-A3B-NVFP4
download_url: https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4
context: 40960
modalities: [text, image, video]
mm_served: false
concurrency: 2
tags: [qwen3.6-35b-a3b, Alibaba, Qwen, NVFP4, 16-40B, Spark recipe, conc-2]
status: done
prefill_toks: 166.18
decode_toks: 147.61
mem_gb: 110.56
mem_source: system MemAvailable delta (10s sampling) over the one-boot conc-2/4/16 sweep — vLLM static KV (util 0.85) + DFlash drafter
spec_acceptance: mean acceptance length ~3.5–4.4 of 11 drafted · avg draft acceptance ~23–31% · per-position ~0.75–0.83 (pos0) decaying to ~0.03–0.08 (pos10)
measured_on: 2026-07-01
completed_at: 2026-07-01 13:33 +0800
engine_image: ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix@sha256:be9e05a11da6e72607ab6f3e960993b253b673af0727005122a3266129a518e3
run_command: |
  # UNTRUSTED image — NO creds; official weights + drafter READ-ONLY; loopback port. ONE boot (max-num-seqs 64,
  # ctx 40960, small-page drafter @31977fbe) sweeping client conc 2/4/16 — matches the published c1/8/32 protocol.
  docker run -d --name aeon-dflash-sweep --gpus all --ipc=host \
    -e TORCH_CUDA_ARCH_LIST=12.1a -e ENABLE_NVFP4_SM100=0 \
    -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -v ~/.cache/huggingface/hub/models--nvidia--Qwen3.6-35B-A3B-NVFP4:/officialmodel:ro \
    -v ~/ornith/drafter-smallpage:/drafter:ro -p 127.0.0.1:8000:8000 \
    --entrypoint vllm ghcr.io/aeon-7/aeon-vllm-ultimate:2026-06-18-v0.23.0-dflashfix \
    serve /officialmodel/snapshots/491c2f1e... --served-model-name official --host 0.0.0.0 --port 8000 \
    --quantization modelopt --moe-backend marlin --trust-remote-code --attention-backend flash_attn \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
    --max-model-len 40960 --gpu-memory-utilization 0.85 --enable-chunked-prefill --enable-prefix-caching \
    --max-num-seqs 64 --max-num-batched-tokens 32768 \
    --speculative-config '{"method":"dflash","model":"/drafter","num_speculative_tokens":11}'
  python3 scripts/bench-serving.py --base-url http://127.0.0.1:8000 --model official \
    --dataset benchmark_data/ShareGPT_V3_unfiltered_cleaned_split.json --concurrency 2 --num-prompts 1000 --max-seconds 600 --max-tokens 256
  # 347/1000 prompts (hit 600 s cap), 0 errors.
---

**conc-2 point of the Qwen3.6-35B-A3B NVFP4 + DFlash line — the crossover already happened.** Official
`nvidia/Qwen3.6-35B-A3B-NVFP4` on the AEON image, DFlash n=11 via the small-page drafter, one-boot sweep,
util 0.85. **0 errors.**

- **Result (conc 2):** prefill 166.18 / decode **147.61** tok/s aggregate; 347/1000 prompts (hit the 600 s
  cap), **0 errors**. **vs MTP conc-2 (161.21): −8.4%.**
- **The sign flips between conc-1 and conc-2.** DFlash *led* single-stream — but only **~+2.9%** (101.9 vs the
  matched 600 s-cap MTP c1 of 99.04; the +8.5% once quoted used a 300 s-cap MTP of 93.9) — and is already
  **behind by conc-2**. The crossover is in the **1→2** region, sharper than the post's "somewhere between 1
  and 8" framing, and the conc-1 lead itself is marginal. The loss then stays a shallow ~−7-8% plateau through conc-8 before collapsing at
  conc-16/32 (see [`-c16`](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash-c16)).
- **Why — wasted draft compute.** DFlash drafts **11** tokens at only **~23–31% acceptance / accept-len
  ~3.5–4.4** → ~7 wasted drafter forward passes per step. Even at conc-2 that extra compute already outweighs
  the bandwidth saving; MTP's 3-token / ~67%-accept / accept-len-3.0 draft has almost no waste. Acceptance is
  workload-driven (flat vs conc), matching the published c1/8/32.
- One server lifetime for c2/c4/c16 → mem is the single 110.56 GB reservation (max-num-seqs 64). TPOT 0.0 is
  the `qwen3` reasoning-parser client artifact — decode tok/s is the real number.
- Series: [`c1/8/32` (main)](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash) ·
  [`-c4`](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash-c4) ·
  [`-c16`](qwen3-6-35b-a3b-nvfp4-vllm-ultimate-dflash-c16). Matched MTP:
  [`-mtp-c2`](qwen3-6-35b-a3b-nvfp4-vllm-mtp-c2).
